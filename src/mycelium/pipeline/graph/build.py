"""
백본 그래프 빌드 (Phase 7.1, LLM 0) — DESIGN_GRAPHRAG §5-1, D-11.

흐름:
  볼트 .md 수집 (ingestion 재사용 + 생성산출물 제외)
  → NoteNode (노트 1개 = 노드, id=source 상대경로)
  → 본문 [[target]] 파싱 → stem 소문자 매칭으로 해소 → links_to 엣지(빈도 가중)
  → frontmatter related → related 엣지
  → 공유 tags → tagged 엣지

생성산출물(graph_report.md·graph.json·.graphify*)은 노이즈 허브라 제외 필수 (D-11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from mycelium.core.artifacts import is_generated_artifact as _is_generated_artifact
from mycelium.core.config import Config
from mycelium.core.models import (
    EDGE_LINKS_TO,
    EDGE_RELATED,
    EDGE_TAGGED,
    NODE_KIND_NOTE,
)
from mycelium.pipeline.ingestion import _collect_md_files, _parse_frontmatter

# 생성산출물 판정은 core.artifacts로 이관(단일 진실, D-11). _is_generated_artifact 별칭은
# 기존 import 호환을 위해 유지(enrich.py 등이 이 이름으로 참조).

# 위키링크 패턴: [[target]] / [[target|별칭]] / [[target#앵커]] 의 target만 캡처.
_WIKILINK_RE = re.compile(r"\[\[([^\]]+?)\]\]")


def _normalize_wikilink_target(target: str) -> str:
    """
    위키링크 target에서 `|별칭`·`#앵커`를 제거하고 stem 소문자로 정규화한다.
    예: "Foo Bar|별칭" → "foo bar", "Note#섹션" → "note", "path/to/Note" → "note".
    """
    # 별칭(|) 제거 — target만 남김
    target = target.split("|", 1)[0]
    # 앵커(#) 제거
    target = target.split("#", 1)[0]
    # 경로 구분자가 있으면 마지막 컴포넌트(stem 후보)만
    target = target.replace("\\", "/").split("/")[-1]
    # .md 확장자 제거 후 소문자
    target = target.strip()
    if target.lower().endswith(".md"):
        target = target[:-3]
    return target.strip().lower()


def _as_list(value: object) -> list[str]:
    """frontmatter 값이 리스트/문자열/None일 수 있어 문자열 리스트로 정규화한다."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    s = str(value).strip()
    return [s] if s else []


def build_backbone_graph(config: Config | None = None) -> nx.Graph:
    """
    위키링크·frontmatter 백본 그래프를 구축해 반환한다 (LLM 0).

    노드: NoteNode (id=source 상대경로, 속성 kind/title/frontmatter type·project·public).
    엣지:
      - links_to: 본문 [[target]] → stem 소문자 매칭으로 해소 (빈도 가중 weight).
      - related : frontmatter related → 같은 방식으로 해소.
      - tagged  : 공유 tag가 있는 노트쌍.

    Returns:
        networkx.Graph (무방향 — 노트 간 연결 구조).
    """
    cfg = config or Config()

    # 1. 노트 수집 (생성산출물 제외) + 메타 파싱.
    #    stem(소문자) → source 매핑, node명(frontmatter node, 소문자) → source 매핑을 함께 만든다.
    #    related는 frontmatter에서 노드명(한글)을 가리키는 경우가 많아 node 필드 매칭이 필요.
    note_sources: list[str] = []  # 그래프에 추가된 노트 source 목록
    stem_to_source: dict[str, str] = {}
    nodename_to_source: dict[str, str] = {}
    note_bodies: dict[str, str] = {}
    note_frontmatter: dict[str, dict] = {}
    note_tags: dict[str, list[str]] = {}

    graph = nx.Graph()

    for md_file in _collect_md_files(cfg.vault_path):
        if _is_generated_artifact(md_file, cfg):
            continue

        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        frontmatter, body = _parse_frontmatter(text)
        source = str(md_file.relative_to(cfg.vault_path))

        note_sources.append(source)
        note_bodies[source] = body
        note_frontmatter[source] = frontmatter

        stem = md_file.stem.lower()
        # 동일 stem이 여러 디렉토리에 있으면 첫 등장만 매핑(나머지는 별도 노드로 존재).
        stem_to_source.setdefault(stem, source)

        # frontmatter node 필드(개념 노드명, 한글일 수 있음)도 매핑 — related 해소용.
        node_name = frontmatter.get("node")
        if node_name:
            nodename_to_source.setdefault(str(node_name).strip().lower(), source)

        tags = [t.lower() for t in _as_list(frontmatter.get("tags"))]
        note_tags[source] = tags

        # NoteNode 추가 (속성: kind/title/frontmatter 핵심 필드).
        title = node_name or md_file.stem
        graph.add_node(
            source,
            kind=NODE_KIND_NOTE,
            title=str(title),
            type=str(frontmatter.get("type", "")),
            project=str(frontmatter.get("project", "")),
            public=bool(frontmatter.get("public", False)),
        )

    def _resolve(target_norm: str) -> str | None:
        """정규화된 링크 타깃을 source로 해소한다 (stem 우선, 없으면 node명 매칭)."""
        if target_norm in stem_to_source:
            return stem_to_source[target_norm]
        if target_norm in nodename_to_source:
            return nodename_to_source[target_norm]
        return None

    unresolved = 0
    resolved_links = 0

    # 2. links_to 엣지 — 본문 위키링크.
    for source in note_sources:
        body = note_bodies[source]
        for raw_target in _WIKILINK_RE.findall(body):
            target_norm = _normalize_wikilink_target(raw_target)
            if not target_norm:
                continue
            dest = _resolve(target_norm)
            if dest is None or dest == source:
                if dest is None:
                    unresolved += 1
                continue
            # 빈도 가중 — 같은 쌍이 여러 번이면 weight 누적.
            if graph.has_edge(source, dest) and graph[source][dest].get("kind") == EDGE_LINKS_TO:
                graph[source][dest]["weight"] += 1.0
            else:
                graph.add_edge(source, dest, kind=EDGE_LINKS_TO, weight=1.0)
            resolved_links += 1

    # 3. related 엣지 — frontmatter related (노드명/파일명 모두 시도).
    for source in note_sources:
        for raw_rel in _as_list(note_frontmatter[source].get("related")):
            rel_norm = raw_rel.strip().lower()
            dest = _resolve(rel_norm)
            if dest is None:
                # 파일명 stem 정규화로 한 번 더 시도(별칭/앵커 제거).
                dest = _resolve(_normalize_wikilink_target(raw_rel))
            if dest is None or dest == source:
                if dest is None:
                    unresolved += 1
                continue
            # links_to가 이미 있으면 그대로 두고(더 강한 신호), 없을 때만 related 추가.
            if not graph.has_edge(source, dest):
                graph.add_edge(source, dest, kind=EDGE_RELATED, weight=1.0)

    # 4. tagged 엣지 — 공유 tag가 있는 노트쌍.
    #    tag → 노트목록 역색인 후, 같은 tag를 공유하는 노트쌍을 연결.
    tag_to_notes: dict[str, list[str]] = {}
    for source, tags in note_tags.items():
        for tag in tags:
            tag_to_notes.setdefault(tag, []).append(source)

    for tag, notes in tag_to_notes.items():
        if len(notes) < 2:
            continue
        # 같은 tag를 공유하는 모든 쌍 연결 (이미 링크/related 있으면 스킵 — 약한 신호).
        for i in range(len(notes)):
            for j in range(i + 1, len(notes)):
                a, b = notes[i], notes[j]
                if not graph.has_edge(a, b):
                    graph.add_edge(a, b, kind=EDGE_TAGGED, weight=1.0)

    # 미해소 링크는 로그로만 (스킵). 진행 가시성.
    if unresolved:
        print(
            f"  [백본] 미해소 링크 {unresolved}건 (대상 노트 없음 — 스킵). "
            f"해소 links_to {resolved_links}건."
        )

    return graph


# ---------------------------------------------------------------------------
# graph-build 오케스트레이션 (7.1 → 7.2 → 7.3)
# ---------------------------------------------------------------------------


@dataclass
class GraphBuildResult:
    """graph-build 결과 집계 (CLI 출력·검증용)."""

    nodes: int
    edges: int
    communities: int
    summaries: int
    modularity: float
    isolated_before: int  # 보강 전(백본만) 고립 노트 수
    isolated_after: int  # 보강 후 고립 노트 수
    graph_path: Path


def _count_isolated_notes(graph: nx.Graph) -> int:
    """degree 0인 NoteNode 수(고립 노트). 엔티티 노드는 제외하고 노트만 센다."""
    count = 0
    for node, data in graph.nodes(data=True):
        if data.get("kind") == NODE_KIND_NOTE and graph.degree(node) == 0:
            count += 1
    return count


def build_graph(config: Config | None = None) -> GraphBuildResult:
    """
    전체 그래프 인덱싱 파이프라인을 실행한다 (7.1 백본 → 7.2 LLM 보강 → 7.3 커뮤니티·요약).

    1. 백본 그래프 구축(위키링크·frontmatter, LLM 0).
    2. 전체 노트 LLM 엔티티·관계 보강(해시캐시).
    3. Leiden 커뮤니티 + LLM 요약(캐시) + 요약 Chroma 적재.
    4. 그래프 영속(gpickle).

    Returns:
        GraphBuildResult — 노드·엣지·커뮤니티·요약 수, modularity, 보강 전후 고립 노트 수.
    """
    from mycelium.adapters.graph_store import save_graph
    from mycelium.pipeline.graph.community import detect_communities
    from mycelium.pipeline.graph.enrich import enrich_graph
    from mycelium.pipeline.graph.summarize import summarize_communities

    cfg = config or Config()

    # 1. 백본 그래프 (LLM 0)
    print("[1/4] 백본 그래프 구축 (위키링크·frontmatter)...")
    graph = build_backbone_graph(cfg)
    isolated_before = _count_isolated_notes(graph)
    print(
        f"  백본: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"고립 노트 {isolated_before}개"
    )

    # 2. LLM 엔티티·관계 보강 (전체 노트, 해시캐시)
    print("[2/4] LLM 엔티티·관계 보강 (전체 노트, 캐시)...")
    enrich_graph(graph, cfg)
    isolated_after = _count_isolated_notes(graph)
    print(f"  보강 후 고립 노트 {isolated_after}개 (보강 전 {isolated_before})")

    # 3. Leiden 커뮤니티 + 요약
    print("[3/4] Leiden 커뮤니티 탐지...")
    n_comm, modularity = detect_communities(graph)
    print(f"  커뮤니티 {n_comm}개, modularity {modularity:.3f}")

    print("[4/4] 커뮤니티 LLM 요약 + Chroma 적재 (캐시)...")
    n_summ = summarize_communities(graph, cfg)
    print(f"  요약 {n_summ}개 적재")

    # 4. 그래프 영속
    path = save_graph(graph, cfg)

    return GraphBuildResult(
        nodes=graph.number_of_nodes(),
        edges=graph.number_of_edges(),
        communities=n_comm,
        summaries=n_summ,
        modularity=modularity,
        isolated_before=isolated_before,
        isolated_after=isolated_after,
        graph_path=path,
    )
