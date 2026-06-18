"""
RAG 답변 생성 파이프라인.

흐름:
  1. HybridRetriever로 관련 청크 검색 (top-k)
  2. 청크들을 [노트] 컨텍스트로 포맷 (source 라벨 포함)
  3. LCEL 체인(prompt | llm | StrOutputParser)으로 LLM 답변 생성
  4. Answer{text, sources} 반환 — 근거 출처는 중복 source 제거 후 RRF 점수 내림차순

할루시네이션 방어 (이중 게이트):
  1차(결정론): 최상위 dense 유사도가 config.relevance_threshold 미만이면 LLM 호출
               전에 no_evidence 확정. LLM 문구 의존의 취약성을 보완.
  2차(보조): system 프롬프트로 "노트 밖 내용 금지" + "근거 없으면 '노트에 근거가
             없습니다'" 유도. LLM이 마커로 답하면 no_evidence.
  둘 중 하나라도 걸리면 no_evidence.
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from mycelium.adapters.llm import create_llm
from mycelium.core.config import Config
from mycelium.core.models import Answer, RetrievedChunk, SourceRef
from mycelium.pipeline.retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# 프롬프트 템플릿 (할루시네이션 방어 포함)
# ---------------------------------------------------------------------------

# 스모크 테스트에서 검증된 프롬프트 패턴 그대로 사용
_SYSTEM_PROMPT = (
    "너는 사용자의 개인 지식베이스를 근거로 답하는 비서다. "
    "아래 [노트]에 있는 내용만 근거로 한국어로 간결히 답하라. "
    "근거가 없으면 '노트에 근거가 없습니다'라고만 답하라."
)

_HUMAN_PROMPT = "[노트]\n{context}\n\n[질문] {question}"

# 근거 없음 마커 — system 프롬프트가 유도하는 정확 문구. agentic.py와 동일.
_NO_EVIDENCE_MARKER = "노트에 근거가 없습니다"

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", _HUMAN_PROMPT),
    ]
)


# ---------------------------------------------------------------------------
# 컨텍스트 포맷 유틸리티
# ---------------------------------------------------------------------------


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """
    검색된 청크 목록을 LLM 컨텍스트 문자열로 조립한다 (혼합 granularity, 7.5).

    일반 청크는 source 경로 라벨, 커뮤니티 요약(kind=community_summary)은 "[커뮤니티 요약]"
    라벨을 붙여 LLM이 출처 단위(개별 노트 vs 주제 군집 요약)를 구분하게 한다.

    형식 예:
      (노트: 03-KNOWLEDGE/foo.md) 청크 본문 전체 텍스트
      [커뮤니티 요약 #3] 군집 요약 텍스트
    """
    parts: list[str] = []
    for chunk in chunks:
        if chunk.kind == "community_summary":
            # 커뮤니티 요약 단위 — 개별 노트가 아니라 주제 군집 조망(혼합 granularity).
            cid = chunk.community_id
            label = f"[커뮤니티 요약 #{cid}]" if cid is not None else "[커뮤니티 요약]"
        else:
            label = f"(노트: {chunk.source})"
        # 전체 본문(text)을 LLM 컨텍스트로 사용 (H1) — 200자 미리보기가 아니라 청크 전문.
        parts.append(f"{label} {chunk.text}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 출처 목록 생성 유틸리티
# ---------------------------------------------------------------------------


def _build_sources(chunks: list[RetrievedChunk]) -> list[SourceRef]:
    """
    검색 청크 목록에서 중복 source를 제거하고 SourceRef 목록을 반환한다.
    같은 source 내 여러 청크가 있으면 가장 높은 RRF 점수를 대표값으로 사용.
    결과는 RRF 점수 내림차순 정렬.

    커뮤니티 요약(kind=community_summary)은 개별 노트가 아니므로 "커뮤니티 N 요약" 라벨로
    출처를 표시한다(혼합 granularity 출처 구분, 7.5).
    """
    # source별 최고 RRF 점수 추적
    best_score: dict[str, float] = {}
    for chunk in chunks:
        if chunk.kind == "community_summary":
            # 요약 단위 출처 라벨 — <community-N> 내부 표기 대신 사람이 읽을 형태.
            src = (
                f"커뮤니티 {chunk.community_id} 요약"
                if chunk.community_id is not None
                else "커뮤니티 요약"
            )
        else:
            src = chunk.source
        if src not in best_score or chunk.rrf_score > best_score[src]:
            best_score[src] = chunk.rrf_score

    # RRF 점수 내림차순 정렬 후 SourceRef 조립 (rank는 1-based)
    sorted_sources = sorted(best_score.items(), key=lambda x: x[1], reverse=True)
    return [
        SourceRef(source=src, rrf_score=score, rank=i + 1)
        for i, (src, score) in enumerate(sorted_sources)
    ]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def generate_answer(
    question: str,
    k: int = 5,
    config: Config | None = None,
    retriever: HybridRetriever | None = None,
) -> Answer:
    """
    질의에 대해 RAG 답변을 생성하고 Answer를 반환한다.

    Parameters
    ----------
    question : str
        사용자 질문.
    k : int
        검색할 상위 청크 수 (기본 5).
    config : Config | None
        설정 객체. None이면 기본 Config() 사용.
    retriever : HybridRetriever | None
        외부에서 주입할 HybridRetriever. None이면 내부 생성.
        MCP 서버처럼 retriever를 재사용하는 경우에 주입.

    Returns
    -------
    Answer
        text: LLM 생성 답변 텍스트
        sources: 근거 출처 노트 목록 (중복 제거, 점수 내림차순)
    """
    cfg = config or Config()

    # 검색기 초기화 (외부 주입이 없으면 내부 생성)
    ret = retriever or HybridRetriever(config=cfg)

    # 1. 하이브리드 검색으로 관련 청크 추출
    chunks = ret.search(question, k=k)

    # 2. 검색 청크가 0개면 LLM 호출 없이 결정론적으로 근거 없음 처리.
    #    빈 컨텍스트로 LLM을 부르면 할루시네이션 위험만 커진다.
    if not chunks:
        return Answer(text=_NO_EVIDENCE_MARKER, sources=[], no_evidence=True)

    # 3. 결정론 게이트 (1차) — 최상위 dense 유사도가 임계 미만이면 LLM 호출 전 근거 없음 확정.
    #    LLM 출력 문구('노트에 근거가 없습니다') 판정은 변형 출력에 취약하므로,
    #    검색 거리(0~1 의미 척도)로 먼저 결정론적으로 걸러낸다. 임계는 골드셋으로 튜닝
    #    (config.relevance_threshold 주석의 측정 근거 참조).
    top_sim = ret.top_dense_similarity(question)
    if top_sim is not None and top_sim < cfg.relevance_threshold:
        return Answer(text=_NO_EVIDENCE_MARKER, sources=[], no_evidence=True)

    # 4. 컨텍스트 포맷 조립
    context = _format_context(chunks)

    # 5. LLM 초기화 및 LCEL 체인 구성
    llm = create_llm(cfg)
    chain = _PROMPT | llm | StrOutputParser()

    # 6. LLM 답변 생성
    answer_text: str = chain.invoke({"context": context, "question": question})

    # 7. 근거 없음 판정 (2차, 보조) — 게이트를 통과했어도 LLM이 마커로 답하면 근거 없음.
    #    거리 게이트(1차, 결정론)와 병행 — 둘 중 하나라도 no_evidence면 no_evidence.
    #    부분 포함(in) 대신 startswith로 좁혀 정상 답변 내 우연한 문구 매칭을 막는다.
    stripped = answer_text.strip()
    if stripped == _NO_EVIDENCE_MARKER or stripped.startswith(_NO_EVIDENCE_MARKER):
        return Answer(text=stripped, sources=[], no_evidence=True)

    # 8. 근거 출처 목록 구성 (중복 source 제거)
    sources = _build_sources(chunks)

    return Answer(text=stripped, sources=sources, no_evidence=False)
