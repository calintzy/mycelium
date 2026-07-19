"""
커뮤니티 LLM 요약 + 임베딩 적재 (Phase 7.3) — DESIGN_GRAPHRAG §5-3·5-4, D-14.

각 커뮤니티의 멤버 노트(제목·핵심)를 ChatOllama로 요약 →
  - graph/community_summaries.json 캐시.
  - 각 요약을 OllamaEmbeddings로 임베딩해 기존 Chroma 벡터스토어에
    metadata={kind:"community_summary", community_id, ...}로 적재(청크와 공존, 후속 검색단위).

캐시: 커뮤니티 멤버 구성 해시 → 요약. 멤버가 같으면 재요약 안 함.
"""

from __future__ import annotations

import hashlib
import json

import networkx as nx

from mycelium.adapters.embedding import create_embeddings
from mycelium.adapters.vectorstore import create_vectorstore
from mycelium.core.config import Config
from mycelium.core.models import NODE_KIND_ENTITY, NODE_KIND_NOTE

# community_summary 메타데이터 kind 값 (Chroma에서 청크와 구분·필터용).
SUMMARY_KIND = "community_summary"

# 요약 캐시 파일명.
_SUMMARY_CACHE = "community_summaries.json"

_SUMMARY_SYSTEM = (
    "너는 지식베이스의 주제 군집을 요약하는 비서다. "
    "아래는 한 커뮤니티(군집)에 속한 노트 제목과 핵심 엔티티 목록이다. "
    "이 군집이 무엇에 관한 것인지 한국어로 2~4문장으로 요약하라. "
    "공통 주제와 핵심 개념을 드러내라."
)

# 요약 프롬프트에 넣을 멤버 노트 제목 최대 수(과대 커뮤니티 컨텍스트 제한).
_MAX_TITLES = 40
_MAX_ENTITIES = 30


def _community_members(graph: nx.Graph) -> dict[int, dict]:
    """
    community_id별로 멤버 노트 제목·엔티티명을 모은다.

    Returns:
        {community_id: {"titles": [...], "entities": [...], "note_count": int}}
    """
    buckets: dict[int, dict] = {}
    for node, data in graph.nodes(data=True):
        cid = data.get("community_id")
        if cid is None:
            continue
        b = buckets.setdefault(cid, {"titles": [], "entities": [], "note_count": 0})
        kind = data.get("kind")
        if kind == NODE_KIND_NOTE:
            b["titles"].append(str(data.get("title") or node))
            b["note_count"] += 1
        elif kind == NODE_KIND_ENTITY:
            b["entities"].append(str(data.get("name") or node))
    return buckets


def _member_hash(titles: list[str], entities: list[str]) -> str:
    """멤버 구성 해시(캐시 키) — 같은 구성이면 재요약 생략."""
    payload = json.dumps(
        {"t": sorted(titles), "e": sorted(entities)}, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_summarizer(config: Config):
    """요약용 ChatOllama (format 미지정 — 자연어 요약)."""
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=config.generation_model,
        base_url=config.ollama_base_url,
        temperature=0,
    )


def _load_summary_cache(config: Config) -> dict:
    """요약 캐시(json)를 로드한다. 없거나 손상이면 빈 dict."""
    path = config.graph_path / _SUMMARY_CACHE
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_summary_cache(config: Config, cache: dict) -> None:
    """요약 캐시(json)를 저장한다."""
    config.graph_path.mkdir(parents=True, exist_ok=True)
    (config.graph_path / _SUMMARY_CACHE).write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarize_communities(graph: nx.Graph, config: Config | None = None) -> int:
    """
    커뮤니티를 LLM 요약하고, 요약을 Chroma에 community_summary 단위로 적재한다.

    - graph/community_summaries.json 캐시(멤버 구성 해시 키).
    - 각 요약을 임베딩해 기존 vault 컬렉션에 kind=community_summary로 적재.
    - 재적재 전 기존 community_summary 항목을 제거해 중복 방지.

    Returns:
        적재된 요약 수.
    """
    cfg = config or Config()
    buckets = _community_members(graph)
    if not buckets:
        return 0

    cache = _load_summary_cache(cfg)
    llm = _make_summarizer(cfg)

    summaries: dict[int, str] = {}  # community_id → 요약 텍스트
    new_cache: dict[str, str] = {}

    for cid, members in sorted(buckets.items()):
        titles = members["titles"][:_MAX_TITLES]
        entities = members["entities"][:_MAX_ENTITIES]
        # 노트 없는 순수 엔티티 군집은 요약 의미가 약해 스킵.
        if members["note_count"] == 0:
            continue

        key = _member_hash(members["titles"], members["entities"])
        if key in cache:
            summary_text = cache[key]
        else:
            prompt = (
                f"[커뮤니티 {cid}]\n"
                f"노트 제목({members['note_count']}개): {', '.join(titles)}\n"
                f"핵심 엔티티: {', '.join(entities) if entities else '없음'}\n\n"
                "이 군집을 2~4문장으로 요약하라."
            )
            try:
                resp = llm.invoke(
                    [("system", _SUMMARY_SYSTEM), ("human", prompt)]
                )
                summary_text = (
                    resp.content
                    if isinstance(resp.content, str)
                    else str(resp.content)
                ).strip()
            except Exception as e:  # noqa: BLE001 — 커뮤니티 1개 실패가 전체를 막지 않게
                print(f"    [경고] 커뮤니티 {cid} 요약 실패, 스킵 — {e}")
                continue

        summaries[cid] = summary_text
        new_cache[key] = summary_text

    # 캐시 저장(이번에 쓰인 것만 — 멤버 구성 바뀐 옛 키는 자연 소멸).
    _save_summary_cache(cfg, new_cache)

    if not summaries:
        return 0

    # --- Chroma 적재: 기존 community_summary 제거 후 재적재(중복 방지) ---
    embeddings = create_embeddings(cfg)
    vectorstore = create_vectorstore(cfg, embeddings)
    collection = vectorstore._collection

    # 기존 community_summary 항목 삭제(있으면).
    try:
        collection.delete(where={"kind": SUMMARY_KIND})
    except Exception:  # noqa: BLE001 — 없으면 무시
        pass

    from langchain_core.documents import Document

    docs: list[Document] = []
    ids: list[str] = []
    for cid, text in summaries.items():
        meta = {
            "kind": SUMMARY_KIND,
            "community_id": int(cid),
            "note_count": int(buckets[cid]["note_count"]),
            "source": f"<community-{cid}>",
            "header_path": "",
            "chunk_id": f"community_summary::{cid}",
        }
        docs.append(Document(page_content=text, metadata=meta))
        ids.append(f"community_summary::{cid}")

    if docs:
        vectorstore.add_documents(documents=docs, ids=ids)

    return len(summaries)
