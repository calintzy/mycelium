"""
LLM 엔티티·관계 보강 (Phase 7.2, 필수) — DESIGN_GRAPHRAG §5-2, D-11.

전체 노트를 노트단위로 ChatOllama(format=json)에 넣어 엔티티·관계를 추출하고,
EntityNode + mentions(노트→엔티티) + relates_to(엔티티↔엔티티) 엣지를 백본 그래프에 더한다.

핵심:
  - 해시캐시: 노트 내용 SHA256 → graph/cache/<sha>.json. 변경 노트만 재추출(재실행 시 캐시 히트).
  - 엔티티명 정규화·중복 병합: 소문자·공백정리한 id로 통합.
  - 실패 노트는 스킵+경고, 전체 중단 금지 (180콜이라 일부 실패 허용).
  - 외부 네트워크 0 (로컬 Ollama).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import networkx as nx

from mycelium.core.config import Config
from mycelium.core.models import (
    EDGE_MENTIONS,
    EDGE_RELATES_TO,
    NODE_KIND_ENTITY,
    NODE_KIND_NOTE,
)
from mycelium.pipeline.graph.build import _is_generated_artifact
from mycelium.pipeline.ingestion import _collect_md_files, _parse_frontmatter

# 추출 프롬프트 — format=json 구조화 출력 유도 (스모크 테스트로 형식 검증됨).
_EXTRACT_SYSTEM = (
    "너는 지식 그래프 추출기다. 주어진 노트에서 핵심 엔티티와 관계를 추출해 "
    "JSON으로만 답하라. 형식: "
    '{"entities": ["엔티티1", "엔티티2"], '
    '"relations": [["엔티티1", "관계", "엔티티2"]]}. '
    "엔티티는 개념·도구·인물·기법 등 핵심 명사(5~15개). "
    "관계는 간결한 동사구. 노트에 명시된 사실만 추출하고 추측하지 마라."
)

# LLM에 넣을 노트 본문 최대 길이(자) — 과대 노트의 컨텍스트·시간 폭주 방지.
_MAX_BODY_CHARS = 6000


def _normalize_entity(name: str) -> str:
    """
    엔티티명을 정규화한다(중복 병합 키).
    소문자화 + 양끝 공백/구두점 제거 + 내부 공백 단일화.
    """
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\n\r.,;:!?'\"()[]{}")
    return s


def _cache_dir(config: Config) -> Path:
    """추출 결과 해시캐시 디렉토리(graph/cache/)를 반환·생성한다."""
    d = config.graph_path / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _content_hash(text: str) -> str:
    """노트 본문 SHA256 해시(캐시 키)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(cache_dir: Path, sha: str) -> dict | None:
    """캐시 히트면 추출 결과 dict, 미스면 None."""
    path = cache_dir / f"{sha}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(cache_dir: Path, sha: str, data: dict) -> None:
    """추출 결과를 캐시에 저장한다."""
    try:
        (cache_dir / f"{sha}.json").write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )
    except OSError:
        pass  # 캐시 저장 실패는 치명적이지 않음(다음 실행에 재추출)


def _make_extractor(config: Config):
    """format=json ChatOllama 추출기를 만든다 (생성 경로와 분리)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.generation_model,
        base_url=config.ollama_base_url,
        temperature=0,
        format="json",
    )


def _extract_one(llm, body: str) -> dict:
    """
    노트 본문 1개에서 엔티티·관계를 추출한다.
    실패 시 빈 결과({"entities": [], "relations": []})를 던지지 않고 예외를 올려보낸다
    (호출측에서 스킵+경고 처리).
    """
    truncated = body[:_MAX_BODY_CHARS]
    messages = [
        ("system", _EXTRACT_SYSTEM),
        ("human", f"[노트]\n{truncated}\n\nJSON으로만 답하라."),
    ]
    resp = llm.invoke(messages)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    data = json.loads(content)
    # 형식 방어 — entities/relations 키가 리스트가 아니면 빈 리스트로.
    entities = data.get("entities") or []
    relations = data.get("relations") or []
    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    return {"entities": entities, "relations": relations}


def enrich_graph(graph: nx.Graph, config: Config | None = None) -> nx.Graph:
    """
    백본 그래프에 LLM 엔티티·관계 보강을 추가한다(전체 노트, in-place 수정 후 반환).

    - EntityNode(id="entity::<정규화명>") + mentions(노트→엔티티) + relates_to(엔티티↔엔티티).
    - 노트 본문 SHA 해시캐시 — 변경 노트만 재추출.
    - 실패 노트는 스킵+경고, 전체 중단 금지.

    Returns:
        보강된 동일 graph 객체.
    """
    cfg = config or Config()
    cache_dir = _cache_dir(cfg)
    llm = _make_extractor(cfg)

    # 보강 대상 노트 = 백본에 NoteNode로 들어간 노트들 (생성산출물 이미 제외됨).
    # 본문은 파일에서 다시 읽는다(백본은 본문을 보관하지 않음).
    note_nodes = [
        n for n, d in graph.nodes(data=True) if d.get("kind") == NODE_KIND_NOTE
    ]
    note_set = set(note_nodes)

    # source(상대경로) → 절대경로 매핑 (본문 재로딩용).
    src_to_path: dict[str, Path] = {}
    for md_file in _collect_md_files(cfg.vault_path):
        if _is_generated_artifact(md_file, cfg):
            continue
        src = str(md_file.relative_to(cfg.vault_path))
        if src in note_set:
            src_to_path.setdefault(src, md_file)

    total = len(note_nodes)
    cache_hits = 0
    extracted = 0
    failures = 0

    # 정규화명 → EntityNode id 매핑(중복 병합). EntityNode id에 "entity::" 접두로
    # NoteNode(source 경로)와 네임스페이스 충돌 방지.
    def _entity_id(name: str) -> str:
        return f"entity::{name}"

    print(f"  [보강] LLM 엔티티 추출 시작 — {total}노트 (수십 분(~49분) 소요, 캐시 후 빨라짐)")

    for i, source in enumerate(note_nodes, start=1):
        path = src_to_path.get(source)
        if path is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            failures += 1
            continue

        _, body = _parse_frontmatter(text)
        if not body.strip():
            continue

        sha = _content_hash(body)
        cached = _load_cache(cache_dir, sha)
        if cached is not None:
            result = cached
            cache_hits += 1
        else:
            try:
                result = _extract_one(llm, body)
                _save_cache(cache_dir, sha, result)
                extracted += 1
            except Exception as e:  # noqa: BLE001 — 노트 1건 실패가 전체를 막지 않게
                failures += 1
                print(f"    [경고] 추출 실패, 스킵: {source} — {type(e).__name__}: {e}")
                continue

        # 진행 로그 — 30노트마다.
        if i % 30 == 0:
            print(
                f"    진행 {i}/{total} (캐시 {cache_hits}, 추출 {extracted}, 실패 {failures})"
            )

        # --- 엔티티 노드 + mentions 엣지 ---
        note_entity_ids: list[str] = []
        for raw_ent in result.get("entities", []):
            name = _normalize_entity(str(raw_ent))
            if not name:
                continue
            eid = _entity_id(name)
            if not graph.has_node(eid):
                graph.add_node(eid, kind=NODE_KIND_ENTITY, name=name)
            note_entity_ids.append(eid)
            # mentions: 노트 → 엔티티 (빈도 가중)
            if graph.has_edge(source, eid):
                graph[source][eid]["weight"] += 1.0
            else:
                graph.add_edge(source, eid, kind=EDGE_MENTIONS, weight=1.0)

        # --- relates_to 엣지 (엔티티 ↔ 엔티티, 관계 라벨) ---
        for rel in result.get("relations", []):
            if not isinstance(rel, (list, tuple)) or len(rel) < 3:
                continue
            e1 = _normalize_entity(str(rel[0]))
            label = str(rel[1]).strip()
            e2 = _normalize_entity(str(rel[2]))
            if not e1 or not e2 or e1 == e2:
                continue
            id1, id2 = _entity_id(e1), _entity_id(e2)
            # 관계 양끝 엔티티 노드 보장(추출 entities에 없던 엔티티도 생성).
            for eid, nm in ((id1, e1), (id2, e2)):
                if not graph.has_node(eid):
                    graph.add_node(eid, kind=NODE_KIND_ENTITY, name=nm)
            if not graph.has_edge(id1, id2):
                graph.add_edge(id1, id2, kind=EDGE_RELATES_TO, label=label, weight=1.0)

    entity_total = sum(
        1 for _, d in graph.nodes(data=True) if d.get("kind") == NODE_KIND_ENTITY
    )
    print(
        f"  [보강] 완료 — 캐시히트 {cache_hits}, 신규추출 {extracted}, 실패 {failures}. "
        f"엔티티 노드 {entity_total}개."
    )

    return graph
