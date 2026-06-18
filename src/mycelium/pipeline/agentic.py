"""
LangGraph 에이전트형 RAG (Phase 6 — 스트레치).

DESIGN.md Phase 6: "검색 부족 시 질의 재작성·재검색".

self-correcting RAG 그래프:
  retrieve → grade → (충분) generate
                   → (불충분) rewrite → retrieve (루프)

  최대 반복 횟수 가드(max_iterations)로 무한루프를 방지한다.

설계:
  - 기존 컴포넌트 재사용(중복 구현 금지):
      HybridRetriever(retrieval.py) — 검색
      create_llm(adapters/llm.py)    — grade·rewrite·generate용 ChatOllama
      generate_answer 프롬프트 패턴   — generate 노드는 동일한 할루시네이션 방어 프롬프트 사용
  - 반환: 답변 + 출처 + 디버그 정보(반복 횟수, 질의 재작성 이력).

레이어: pipeline → adapters → core (의존 방향 준수).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

from mycelium.adapters.llm import create_llm
from mycelium.core.config import Config
from mycelium.core.models import Answer, RetrievedChunk, SourceRef
from mycelium.pipeline.generation import _build_sources, _format_context
from mycelium.pipeline.retrieval import HybridRetriever

# ---------------------------------------------------------------------------
# 프롬프트 — grade(관련성 판정), rewrite(질의 재작성), generate(답변 생성)
# ---------------------------------------------------------------------------

# grade: 검색된 청크가 질문에 충분히 관련 있는지 yes/no 1단어로만 판정.
_GRADE_SYSTEM = (
    "너는 검색 품질 평가자다. 아래 [검색결과]가 [질문]에 답하기에 "
    "충분히 관련 있는지 판정하라. 'yes' 또는 'no' 한 단어로만 답하라. "
    "다른 말은 절대 덧붙이지 마라."
)
_GRADE_HUMAN = "[질문] {question}\n\n[검색결과]\n{context}"
_GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _GRADE_SYSTEM), ("human", _GRADE_HUMAN)]
)

# rewrite: 검색이 불충분할 때 질의를 더 검색 친화적으로 재작성.
_REWRITE_SYSTEM = (
    "너는 검색 질의 최적화 전문가다. 사용자의 [원본질문]이 모호하거나 "
    "검색이 잘 안 됐다. 개인 지식베이스 검색에 더 잘 맞도록 핵심 키워드를 "
    "살린 질의문 한 줄로 재작성하라. 설명 없이 재작성된 질의만 출력하라."
)
_REWRITE_HUMAN = "[원본질문] {question}"
_REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _REWRITE_SYSTEM), ("human", _REWRITE_HUMAN)]
)

# generate: generation.py와 동일한 할루시네이션 방어 프롬프트(서사 일관성).
_GENERATE_SYSTEM = (
    "너는 사용자의 개인 지식베이스를 근거로 답하는 비서다. "
    "아래 [노트]에 있는 내용만 근거로 한국어로 간결히 답하라. "
    "근거가 없으면 '노트에 근거가 없습니다'라고만 답하라."
)
_GENERATE_HUMAN = "[노트]\n{context}\n\n[질문] {question}"
_GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _GENERATE_SYSTEM), ("human", _GENERATE_HUMAN)]
)

# generation.py와 동일한 근거 없음 마커.
_NO_EVIDENCE_MARKER = "노트에 근거가 없습니다"


# ---------------------------------------------------------------------------
# 그래프 State
# ---------------------------------------------------------------------------


class _AgentState(TypedDict):
    """
    LangGraph StateGraph가 노드 간 전달하는 상태.

    - original_query: 사용자 원본 질의(불변, 디버그·출처 표시용).
    - current_query: 현재 검색에 사용 중인 질의(rewrite로 갱신됨).
    - chunks: 마지막 retrieve가 반환한 청크.
    - iterations: retrieve 실행 횟수(무한루프 가드 기준).
    - rewrites: 질의 재작성 이력(에이전트 동작 가시화용).
    - answer_text: generate 노드가 채우는 최종 답변.
    - grade_sufficient: grade 노드가 1회 판정한 관련성 충분 여부(_decide가 읽기만 함).
    """

    original_query: str
    current_query: str
    chunks: list[RetrievedChunk]
    iterations: int
    rewrites: list[str]
    answer_text: str
    grade_sufficient: bool


# ---------------------------------------------------------------------------
# 반환 결과 — Answer + 에이전트 디버그 정보
# ---------------------------------------------------------------------------


@dataclass
class AgenticResult:
    """
    에이전트형 RAG 실행 결과.

    - answer: 최종 RAG 답변(text + sources + no_evidence).
    - iterations: 실제 검색 반복 횟수(1=루프 없음, >1=재검색 발생).
    - rewrites: 질의 재작성 이력(원본→재작성). 비어 있으면 1회 검색으로 충분했음.
    - max_iterations_hit: 최대 반복 가드에 걸려 멈췄으면 True.
    """

    answer: Answer
    iterations: int
    rewrites: list[str] = field(default_factory=list)
    max_iterations_hit: bool = False


# ---------------------------------------------------------------------------
# 에이전트 그래프 빌더
# ---------------------------------------------------------------------------


class AgenticRAG:
    """
    LangGraph self-correcting RAG.

    검색이 불충분하면 질의를 재작성해 재검색하고, 충분하면 답변을 생성한다.
    HybridRetriever·LLM은 1회 초기화 후 재사용한다(반복 검색마다 재구축 없음).

    Parameters
    ----------
    config : Config | None
        프로젝트 설정. None이면 기본 Config().
    k : int
        검색할 청크 수(기본 5).
    max_iterations : int
        retrieve 최대 반복 횟수. 기본 2 — 초기 검색 1 + 재검색 1.
        이 횟수에 도달하면 grade 결과와 무관하게 generate로 진행(무한루프 방지).
    retriever : HybridRetriever | None
        외부 주입 검색기. None이면 내부 생성.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        k: int = 5,
        max_iterations: int = 2,
        retriever: Optional[HybridRetriever] = None,
    ) -> None:
        self._config = config or Config()
        self._k = k
        self._max_iterations = max(1, max_iterations)
        self._retriever = retriever or HybridRetriever(config=self._config)
        self._llm = create_llm(self._config)
        # LCEL 체인 — 노드마다 재사용
        self._grade_chain = _GRADE_PROMPT | self._llm | StrOutputParser()
        self._rewrite_chain = _REWRITE_PROMPT | self._llm | StrOutputParser()
        self._generate_chain = _GENERATE_PROMPT | self._llm | StrOutputParser()
        self._graph = self._build_graph()

    # --- 노드 정의 ---------------------------------------------------------

    def _retrieve(self, state: _AgentState) -> dict:
        """현재 질의로 하이브리드 검색을 실행하고 반복 횟수를 증가시킨다."""
        chunks = self._retriever.search(state["current_query"], k=self._k)
        return {
            "chunks": chunks,
            "iterations": state["iterations"] + 1,
        }

    def _grade(self, state: _AgentState) -> dict:
        """
        검색 결과의 관련성을 LLM으로 1회 판정해 state에 저장한다.
        실제 분기는 _decide()가 이 state 값만 읽어 결정한다(LLM 중복 호출 제거).
        청크가 없으면 LLM 호출 없이 불충분으로 둔다(_decide가 generate로 보냄).
        """
        if not state["chunks"]:
            return {"grade_sufficient": False}

        context = _format_context(state["chunks"])
        verdict = self._grade_chain.invoke(
            {"question": state["original_query"], "context": context}
        )
        # 'yes'가 포함되면 충분으로 판정(LLM이 'Yes.' 등으로 답할 수 있음).
        sufficient = "yes" in verdict.strip().lower()
        return {"grade_sufficient": sufficient}

    def _rewrite(self, state: _AgentState) -> dict:
        """검색이 불충분할 때 LLM으로 질의를 재작성하고 이력에 기록한다."""
        new_query = self._rewrite_chain.invoke(
            {"question": state["original_query"]}
        ).strip()
        # 재작성 이력에 누적(원본 → 재작성 가시화)
        rewrites = list(state["rewrites"])
        rewrites.append(new_query)
        return {"current_query": new_query, "rewrites": rewrites}

    def _generate(self, state: _AgentState) -> dict:
        """충분한 청크로 최종 답변을 생성한다(generation.py와 동일 프롬프트)."""
        context = _format_context(state["chunks"])
        answer_text = self._generate_chain.invoke(
            {"context": context, "question": state["original_query"]}
        ).strip()
        return {"answer_text": answer_text}

    # --- 조건부 엣지 -------------------------------------------------------

    def _decide(self, state: _AgentState) -> str:
        """
        grade 후 분기 결정 — state["grade_sufficient"]만 읽는다(LLM 재호출 없음).
        - 최대 반복 도달 → 'generate' (무한루프 가드, 반드시).
        - 청크 없음 → 'generate' (재작성해도 의미 없음, no_evidence로 귀결).
        - grade 충분 → 'generate'.
        - 그 외(불충분) → 'rewrite' (재작성 후 재검색).
        """
        # 무한루프 방지 가드 — grade 결과보다 우선.
        if state["iterations"] >= self._max_iterations:
            return "generate"
        if not state["chunks"]:
            return "generate"
        if state["grade_sufficient"]:
            return "generate"
        return "rewrite"

    # --- 그래프 조립 -------------------------------------------------------

    def _build_graph(self):
        """StateGraph를 조립해 컴파일된 그래프를 반환한다."""
        graph = StateGraph(_AgentState)

        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("generate", self._generate)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        # grade 후 조건부 분기: 충분→generate, 불충분→rewrite
        graph.add_conditional_edges(
            "grade",
            self._decide,
            {"generate": "generate", "rewrite": "rewrite"},
        )
        # rewrite 후 다시 retrieve로 루프
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    # --- 공개 API ----------------------------------------------------------

    def run(self, question: str) -> AgenticResult:
        """
        에이전트형 RAG를 실행하고 결과를 반환한다.

        Parameters
        ----------
        question : str
            사용자 질문.

        Returns
        -------
        AgenticResult
            답변 + 반복 횟수 + 질의 재작성 이력 + 최대반복 도달 여부.
        """
        initial: _AgentState = {
            "original_query": question,
            "current_query": question,
            "chunks": [],
            "iterations": 0,
            "rewrites": [],
            "answer_text": "",
            "grade_sufficient": False,
        }
        final = self._graph.invoke(initial)

        # 최종 답변 조립 (generation.py 규칙과 동일하게 근거 없음 처리).
        # 청크 0개이거나 답변이 마커로 시작하면 근거 없음 — 부분 포함(in) 대신 좁힌다.
        answer_text = final["answer_text"].strip()
        chunks = final["chunks"]
        if (
            not chunks
            or answer_text == _NO_EVIDENCE_MARKER
            or answer_text.startswith(_NO_EVIDENCE_MARKER)
        ):
            answer = Answer(text=answer_text or _NO_EVIDENCE_MARKER, sources=[], no_evidence=True)
        else:
            sources: list[SourceRef] = _build_sources(chunks)
            answer = Answer(text=answer_text, sources=sources, no_evidence=False)

        max_hit = final["iterations"] >= self._max_iterations and bool(
            final["rewrites"]
        )
        return AgenticResult(
            answer=answer,
            iterations=final["iterations"],
            rewrites=final["rewrites"],
            max_iterations_hit=max_hit,
        )


def run_agentic(
    question: str,
    k: int = 5,
    max_iterations: int = 2,
    config: Optional[Config] = None,
) -> AgenticResult:
    """
    에이전트형 RAG 1회 실행 편의 함수 (CLI 진입점용).
    내부에서 AgenticRAG를 생성해 run()을 호출한다.
    """
    agent = AgenticRAG(config=config, k=k, max_iterations=max_iterations)
    return agent.run(question)
