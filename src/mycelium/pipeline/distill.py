"""
distill — Q&A → 큐레이션 위키노트 (Phase 7.7, 컴파운딩) — DESIGN_GRAPHRAG §7, D-15.

흐름:
  1. 주제/질문으로 통합검색(HybridRetriever) → 관련 근거 청크 수집.
  2. LLM이 큐레이션 위키노트 본문 작성(헤더 구조). 근거 청크만 사용(할루시네이션 방어).
  3. 기존 그래프 노드(노트)에서 고른 `[[관련노트]]` 링크를 덧붙인다 — 실재 노트만(검증).
  4. frontmatter(type:wiki, public:false deny-by-default, date) + 본문 조립.
  5. wiki_dir에 저장:
     - 자동 덮어쓰기 금지. 동일 파일명 존재 시 `.new` 접미로 별도 저장(중단 안내).
     - 기본 --dry-run(미리보기만), --write일 때만 실제 저장.
  6. 저장 후 증분 인덱싱: 그 노트의 청크만 Chroma에 add(upsert).
     ⚠️ 전체 재인덱싱(_reset) 호출 금지 — 같은 컬렉션의 커뮤니티 요약 41개가 날아간다.
     그래프 증분은 어려우므로 "graph-build 재실행 필요" 안내로 대체(청크 인덱싱은 증분).

외부 네트워크 0 (로컬 Ollama). deny-by-default 프라이버시(public:false).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

from mycelium.adapters.embedding import create_embeddings
from mycelium.adapters.vectorstore import create_vectorstore
from mycelium.core.config import Config
from mycelium.core.models import NODE_KIND_NOTE, RetrievedChunk
from mycelium.pipeline.ingestion import _split_document, _parse_frontmatter
from mycelium.pipeline.retrieval import HybridRetriever

# 위키노트 본문 작성 system 프롬프트 — 근거 청크만 사용(할루시네이션 방어).
_DISTILL_SYSTEM = (
    "너는 개인 지식베이스를 큐레이션하는 위키 편집자다. "
    "아래 [근거]에 있는 내용만 사용해, 주어진 주제에 대한 위키노트 본문을 한국어로 작성하라. "
    "마크다운 헤더(##, ###)로 구조화하고, 핵심을 간결히 정리하라. "
    "근거에 없는 내용을 지어내지 마라. frontmatter나 제목(# 한 개짜리)은 쓰지 말고 본문만 써라."
)

# 관련 노트 링크 선별 프롬프트 — 후보 노트 제목 목록에서 실제 관련된 것만 고르게 한다.
_LINK_SYSTEM = (
    "너는 위키 편집자다. 아래 [주제]와 [후보 노트] 목록을 보고, 이 위키노트에 "
    "`[[관련노트]]` 링크로 연결하기 적절한 노트 제목만 골라 쉼표로 구분해 나열하라. "
    "반드시 후보 목록에 있는 제목만 그대로 쓰고, 없는 제목은 만들지 마라. "
    "적절한 것이 없으면 빈 줄로 답하라."
)

# 위키노트 frontmatter date 미지정 시 본문 작성 컨텍스트로 넣을 근거 청크 최대 수.
_MAX_EVIDENCE = 6
# 링크 후보로 LLM에 제시할 노트 제목 최대 수(컨텍스트 폭주 방지).
_MAX_LINK_CANDIDATES = 30


@dataclass
class DistillResult:
    """distill 결과 집계 (CLI 출력·검증용)."""

    filename: str  # 저장(또는 미리보기) 대상 파일명
    target_path: Path  # 실제 저장 경로(.new 접미 포함될 수 있음)
    note_text: str  # 생성된 위키노트 전문(frontmatter + 본문)
    links: list[str] = field(default_factory=list)  # 채택된 [[관련노트]] (실재 노트만)
    evidence_sources: list[str] = field(default_factory=list)  # 근거 노트 source 목록
    written: bool = False  # 실제 저장 여부(--write)
    skipped_existing: bool = False  # 동일 파일명 존재로 .new 접미 저장했는지
    indexed_chunks: int = 0  # 증분 인덱싱된 청크 수
    no_evidence: bool = False  # 근거 없음(검색 0건)


def _slugify(topic: str) -> str:
    """주제 문자열을 파일명 stem으로 정규화한다(공백→하이픈, 특수문자 제거)."""
    s = topic.strip()
    # 파일명에 위험한 문자 제거(경로구분자·제어문자 등), 공백은 하이픈.
    s = re.sub(r"[\\/:*?\"<>|]+", "", s)
    s = re.sub(r"\s+", "-", s)
    s = s.strip("-")
    return s or "wiki-note"


def _note_title_candidates(config: Config) -> dict[str, str]:
    """
    기존 그래프 노트 노드에서 링크 후보 {제목소문자: source} 매핑을 만든다.

    그래프가 있으면 NoteNode의 title을, 없으면 빈 dict.
    링크 검증·해소에 쓴다 — LLM이 고른 제목이 실재 노트인지 이 맵으로 확인한다.
    """
    try:
        from mycelium.adapters.graph_store import graph_exists, load_graph
    except Exception:  # noqa: BLE001
        return {}

    if not graph_exists(config):
        return {}
    try:
        graph = load_graph(config)
    except Exception:  # noqa: BLE001 — 그래프 로드 실패는 링크 없이 진행
        return {}

    candidates: dict[str, str] = {}
    for node, data in graph.nodes(data=True):
        if data.get("kind") != NODE_KIND_NOTE:
            continue
        title = str(data.get("title") or "").strip()
        if not title:
            # title 없으면 파일 stem으로 폴백.
            title = Path(str(node)).stem
        candidates.setdefault(title.lower(), str(node))
    return candidates


def _select_links(
    llm, topic: str, candidates: dict[str, str], exclude_stem: str
) -> list[str]:
    """
    LLM에게 후보 노트 제목 중 관련된 것을 고르게 하고, 실재하는 제목만 [[링크]]로 반환한다.

    - candidates: {제목소문자: source}. LLM이 고른 제목이 이 키에 있어야 채택(실재 검증).
    - exclude_stem: 지금 만드는 노트 자신은 자기링크 방지로 제외.
    """
    if not candidates:
        return []

    # 제목 원형 목록(LLM 제시용) — 소문자 키에서 표시용은 별도 보관이 없으므로 키 그대로 쓴다.
    # candidates 키는 소문자지만 비교도 소문자로 하므로 일관.
    titles = list(candidates.keys())[:_MAX_LINK_CANDIDATES]
    prompt = (
        f"[주제] {topic}\n\n"
        f"[후보 노트]\n{', '.join(titles)}\n\n"
        "관련된 제목만 쉼표로 구분해 나열하라(없으면 빈 줄)."
    )
    try:
        resp = llm.invoke([("system", _LINK_SYSTEM), ("human", prompt)])
        raw = (
            resp.content if isinstance(resp.content, str) else str(resp.content)
        ).strip()
    except Exception:  # noqa: BLE001 — 링크 선별 실패는 링크 없이 진행
        return []

    links: list[str] = []
    seen: set[str] = set()
    for piece in raw.replace("\n", ",").split(","):
        cand = piece.strip().lower()
        if not cand or cand in seen:
            continue
        # 실재 노트만 채택(할루시네이션 링크 차단). 자기 자신 제외.
        if cand in candidates and cand != exclude_stem.lower():
            seen.add(cand)
            links.append(cand)
    return links


def _gather_evidence(
    retriever: HybridRetriever, topic: str, k: int
) -> list[RetrievedChunk]:
    """통합검색으로 근거 청크를 모은다(커뮤니티 요약 포함될 수 있음)."""
    return retriever.search(topic, k=k)


def _compose_note(
    topic: str,
    body: str,
    links: list[str],
    note_date: str | None,
) -> str:
    """frontmatter + 본문 + [[관련노트]] 섹션을 조립해 위키노트 전문을 만든다."""
    # frontmatter — deny-by-default(public:false), type:wiki, date(인자로 받거나 생략).
    fm_lines = ["---", "type: wiki", "public: false"]
    if note_date:
        fm_lines.append(f"date: {note_date}")
    fm_lines.append("---")
    frontmatter = "\n".join(fm_lines)

    parts = [frontmatter, "", f"# {topic}", "", body.strip()]

    if links:
        parts.append("")
        parts.append("## 관련 노트")
        parts.append("")
        # 실재 노트만 링크(검증 통과한 것). Obsidian 위키링크 형식.
        parts.extend(f"- [[{link}]]" for link in links)

    return "\n".join(parts).rstrip() + "\n"


def _incremental_index(
    config: Config,
    target_path: Path,
    note_text: str,
) -> int:
    """
    저장된 위키노트의 청크만 Chroma에 add한다 (증분, _reset 호출 금지).

    ⚠️ 전체 재인덱싱을 하면 같은 컬렉션의 커뮤니티 요약 41개가 날아가므로,
    이 노트의 청크만 _split_document로 만들어 add_documents로 추가(upsert)한다.

    Returns:
        추가된 청크 수.
    """
    frontmatter, body = _parse_frontmatter(note_text)
    docs = _split_document(
        file_path=target_path,
        vault_path=config.vault_path,
        frontmatter=frontmatter,
        body=body,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    if not docs:
        return 0

    embeddings = create_embeddings(config)
    vectorstore = create_vectorstore(config, embeddings)

    # chunk_id를 문서 ID로 사용 — 재distill 시 같은 id는 upsert(중복 누적 방지).
    ids = [d.metadata.get("chunk_id") or "" for d in docs]
    # 빈 id 방어(이론상 _split_document가 항상 chunk_id 부여).
    # 파일명(동명 충돌 위험) 대신 볼트 기준 상대경로를 폴백 키로 사용.
    try:
        rel = str(target_path.relative_to(config.vault_path))
    except ValueError:
        rel = target_path.name
    ids = [cid if cid else f"{rel}::{i}" for i, cid in enumerate(ids)]

    vectorstore.add_documents(documents=docs, ids=ids)
    return len(docs)


def distill(
    topic: str,
    config: Config | None = None,
    k: int = 8,
    note_date: str | None = None,
    write: bool = False,
) -> DistillResult:
    """
    주제/질문을 큐레이션 위키노트로 정제한다 (Phase 7.7).

    Parameters
    ----------
    topic : str
        주제 또는 질문.
    config : Config | None
        설정 객체. None이면 기본 Config().
    k : int
        통합검색 근거 청크 수(기본 8).
    note_date : str | None
        frontmatter date 값. None이면 date 필드 생략(인자로 받거나 생략, D-15).
        "today"면 오늘 날짜를 채운다.
    write : bool
        True면 실제 저장 + 증분 인덱싱. False(기본)면 미리보기만(--dry-run).

    Returns
    -------
    DistillResult
        생성된 위키노트 전문·링크·근거·저장 여부·증분 인덱싱 수.
    """
    cfg = config or Config()

    # "today" 편의값 처리(인자 없으면 None 유지 = date 생략).
    if note_date == "today":
        note_date = _date.today().isoformat()

    # 1. 통합검색으로 근거 수집.
    retriever = HybridRetriever(config=cfg)
    if not retriever.has_corpus():
        # 인덱스 비어 있으면 근거 없음.
        stem = _slugify(topic)
        return DistillResult(
            filename=f"{stem}.md",
            target_path=cfg.vault_path / cfg.wiki_dir / f"{stem}.md",
            note_text="",
            no_evidence=True,
        )

    evidence = _gather_evidence(retriever, topic, k=k)
    if not evidence:
        stem = _slugify(topic)
        return DistillResult(
            filename=f"{stem}.md",
            target_path=cfg.vault_path / cfg.wiki_dir / f"{stem}.md",
            note_text="",
            no_evidence=True,
        )

    # 근거 컨텍스트 조립(전문 사용, 상위 _MAX_EVIDENCE개).
    from mycelium.adapters.llm import create_llm

    llm = create_llm(cfg)
    top_evidence = evidence[:_MAX_EVIDENCE]
    evidence_ctx = "\n\n".join(
        f"(노트: {c.source}) {c.text}" if c.kind != "community_summary"
        else f"[커뮤니티 요약] {c.text}"
        for c in top_evidence
    )

    # 2. LLM 본문 작성(근거만 사용).
    body_resp = llm.invoke(
        [
            ("system", _DISTILL_SYSTEM),
            ("human", f"[주제] {topic}\n\n[근거]\n{evidence_ctx}\n\n위키노트 본문을 써라."),
        ]
    )
    body = (
        body_resp.content
        if isinstance(body_resp.content, str)
        else str(body_resp.content)
    ).strip()

    # 3. 기존 그래프 노드에서 [[관련노트]] 링크 선별(실재 노트만).
    stem = _slugify(topic)
    candidates = _note_title_candidates(cfg)
    # 자기링크 가드: candidates 키는 제목 소문자 기준이므로 slug가 아닌 topic 소문자로 비교.
    # slug(하이픈 정규화)와 제목 소문자(공백 유지)가 달라 slug로 비교하면 가드가 무력화됨.
    links = _select_links(llm, topic, candidates, exclude_stem=topic.lower())

    # 4. frontmatter + 본문 + 링크 조립.
    note_text = _compose_note(topic, body, links, note_date)

    # 근거 노트 source 목록(요약 제외, 중복 제거).
    ev_sources: list[str] = []
    seen_src: set[str] = set()
    for c in top_evidence:
        if c.kind == "community_summary":
            continue
        if c.source and c.source not in seen_src:
            seen_src.add(c.source)
            ev_sources.append(c.source)

    # 5. 저장 경로 결정 — 덮어쓰기 금지.
    wiki_dir = cfg.vault_path / cfg.wiki_dir
    base_path = wiki_dir / f"{stem}.md"
    target_path = base_path
    skipped_existing = False
    if base_path.exists():
        # 자동 덮어쓰기 금지 — .new 접미로 분리 저장(중단 대신 보존).
        target_path = wiki_dir / f"{stem}.new.md"
        skipped_existing = True

    result = DistillResult(
        filename=base_path.name,
        target_path=target_path,
        note_text=note_text,
        links=links,
        evidence_sources=ev_sources,
        written=False,
        skipped_existing=skipped_existing,
        indexed_chunks=0,
        no_evidence=False,
    )

    # 6. --dry-run(기본)이면 여기서 미리보기만 반환.
    if not write:
        return result

    # --write — 실제 저장 + 증분 인덱싱.
    wiki_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(note_text, encoding="utf-8")
    result.written = True

    # 증분 인덱싱(이 노트 청크만 add — _reset 금지, 커뮤니티 요약 보존).
    result.indexed_chunks = _incremental_index(cfg, target_path, note_text)

    return result
