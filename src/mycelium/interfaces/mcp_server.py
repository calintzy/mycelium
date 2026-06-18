"""
MCP 서버 인터페이스 — FastMCP 기반 mycelium 툴 노출.

Claude Code가 마크다운 볼트를 의미검색(vault_search)하거나
RAG 답변을 생성(vault_ask)하도록 MCP 프로토콜로 노출한다.

설계 결정:
  - 무거운 객체(HybridRetriever, BM25 인덱스, 임베딩 모델, LLM)는
    서버 기동 시 1회 초기화(모듈 레벨 lazy 싱글턴)해 매 툴 호출마다 재구축하지 않는다.
  - 레이어 의존 방향: interfaces → pipeline → adapters → core
  - stdio 트랜스포트로 실행 (MCP 표준 방식)
"""

from __future__ import annotations

import json
import logging

from mcp.server.fastmcp import FastMCP

from mycelium.core.config import Config
from mycelium.core.models import Answer, RetrievedChunk

# ---------------------------------------------------------------------------
# 로거 설정
# ---------------------------------------------------------------------------

logger = logging.getLogger("mycelium.mcp")

# ---------------------------------------------------------------------------
# FastMCP 서버 인스턴스
# ---------------------------------------------------------------------------

mcp = FastMCP("mycelium")

# ---------------------------------------------------------------------------
# 무거운 객체 — 모듈 레벨 lazy 싱글턴 (서버 기동 시 1회 초기화)
# ---------------------------------------------------------------------------

_config: Config | None = None
_retriever = None  # HybridRetriever 인스턴스 (타입 순환 방지로 Any)


def _get_retriever():
    """
    HybridRetriever 싱글턴을 반환한다.
    최초 호출 시 Chroma 로드 + BM25 인덱스 구축을 1회 실행한다.
    이후 호출에서는 기구축된 인스턴스를 즉시 반환한다.
    """
    global _config, _retriever

    if _retriever is not None:
        return _retriever

    # Config 초기화 (환경변수 존중)
    _config = Config()

    logger.info("HybridRetriever 초기화 중 (BM25 인덱스 구축 포함)...")
    from mycelium.pipeline.retrieval import HybridRetriever

    _retriever = HybridRetriever(config=_config)
    logger.info("HybridRetriever 초기화 완료.")

    return _retriever


# ---------------------------------------------------------------------------
# MCP 툴 정의
# ---------------------------------------------------------------------------


@mcp.tool()
def vault_search(query: str, k: int = 5) -> str:
    """
    볼트를 하이브리드 검색(dense 의미검색 + BM25 키워드 + RRF 융합)으로 검색한다.

    언제 호출하는가:
      - 사용자가 볼트에서 특정 주제·키워드와 관련된 노트를 찾으려 할 때.
      - 질문에 답하기 전에 관련 노트를 먼저 확인하고 싶을 때.
      - 요약·비교·분석 없이 관련 청크 목록 자체를 원할 때.

    반환값:
      JSON 문자열. 각 항목에 순위(rank), 출처 파일(source), 헤더 경로(header_path),
      본문 미리보기(preview, 200자), RRF 융합 점수(rrf_score)가 포함된다.
      결과는 rrf_score 내림차순(관련성 높은 순).

    Parameters
    ----------
    query : str
        검색 질의 (한국어/영어 모두 가능).
    k : int
        반환할 결과 수 (기본 5, 최대 50).
    """
    k = max(1, min(k, 50))  # 범위 방어

    try:
        retriever = _get_retriever()
        # 빈 코퍼스 안내 (H4 UX) — 인덱싱 전이면 검색해도 결과가 없으므로 명확히 안내.
        if not retriever.has_corpus():
            return json.dumps(
                {
                    "query": query,
                    "error": "인덱스가 비어 있습니다. 먼저 index를 실행하세요.",
                    "results": [],
                    "total": 0,
                },
                ensure_ascii=False,
            )
        chunks: list[RetrievedChunk] = retriever.search(query, k=k)
    except Exception as exc:
        logger.error("vault_search 실행 오류: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    if not chunks:
        return json.dumps(
            {"query": query, "k": k, "results": [], "total": 0},
            ensure_ascii=False,
        )

    results = []
    for rank, chunk in enumerate(chunks, start=1):
        results.append(
            {
                "rank": rank,
                "source": chunk.source,
                "header_path": chunk.header_path,
                "preview": chunk.text_preview,
                "rrf_score": round(chunk.rrf_score, 6),
                "dense_rank": chunk.dense_rank,
                "bm25_rank": chunk.bm25_rank,
            }
        )

    return json.dumps(
        {"query": query, "k": k, "results": results, "total": len(results)},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def vault_ask(query: str, k: int = 5) -> str:
    """
    볼트를 근거로 RAG 답변을 생성한다.

    언제 호출하는가:
      - 사용자가 볼트 내용을 바탕으로 구체적인 질문에 답을 원할 때.
      - 단순 검색이 아니라 노트를 종합해 자연어 답변이 필요할 때.
      - "내 볼트에 X에 대한 내용이 있어?" 또는 "Y에 대해 정리해줘" 류 요청 시.

    반환값:
      JSON 문자열. answer(LLM 생성 한국어 답변), no_evidence(근거 없음 여부),
      sources(근거 노트 목록, 중복 source 제거·RRF 점수 내림차순) 포함.
      no_evidence=true이면 볼트에 관련 내용이 없는 것이므로 sources는 빈 배열.

    Parameters
    ----------
    query : str
        질문 (한국어/영어 모두 가능).
    k : int
        검색에 사용할 청크 수 (기본 5, 최대 50).
    """
    k = max(1, min(k, 50))  # 범위 방어

    try:
        retriever = _get_retriever()
        cfg = _config or Config()

        # 빈 코퍼스 안내 (H4 UX) — 인덱싱 전이면 LLM 호출 전에 명확히 안내.
        if not retriever.has_corpus():
            return json.dumps(
                {
                    "query": query,
                    "answer": "인덱스가 비어 있습니다. 먼저 index를 실행하세요.",
                    "no_evidence": True,
                    "sources": [],
                },
                ensure_ascii=False,
            )

        from mycelium.pipeline.generation import generate_answer

        answer: Answer = generate_answer(
            question=query,
            k=k,
            config=cfg,
            retriever=retriever,  # 기구축된 retriever 재사용 (재초기화 금지)
        )
    except Exception as exc:
        logger.error("vault_ask 실행 오류: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    sources = []
    if not answer.no_evidence:
        for ref in answer.sources:
            sources.append(
                {
                    "rank": ref.rank,
                    "source": ref.source,
                    "rrf_score": round(ref.rrf_score, 6),
                }
            )

    return json.dumps(
        {
            "query": query,
            "answer": answer.text,
            "no_evidence": answer.no_evidence,
            "sources": sources,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# 직접 실행 진입점 (python -m mycelium serve 또는 python mcp_server.py)
# ---------------------------------------------------------------------------


def run_server() -> None:
    """MCP 서버를 stdio 트랜스포트로 실행한다."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    logger.info("mycelium MCP 서버 기동 중 (stdio)...")
    mcp.run()


if __name__ == "__main__":
    run_server()
