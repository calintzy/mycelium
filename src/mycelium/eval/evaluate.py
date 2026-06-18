"""
검색 품질 평가 하니스 — grep / 의미검색 / 하이브리드 3방식 비교 (Phase 4, DESIGN §7)
+ 평면(2신호) vs 통합(4신호) 그래프 비교 + 가중치 데이터튜닝 (Phase 7.8, DESIGN_GRAPHRAG §8).

평가 방식:
  1) grep 베이스라인 (현 워크플로 대표):
     질문을 공백 분리한 단어들로 볼트 .md 본문을 매칭, 파일별 매칭 단어 수
     내림차순 정렬. 한국어 조사 미분리로 약할 것이며, 그게 비교의 포인트다.
     → 순위 개념이 없는 집합 방식이므로 MRR은 정의되지 않음(N/A).
  2) 의미검색 단독:
     HybridRetriever.dense_search() — Chroma similarity_search top-k.
  3) 하이브리드:
     HybridRetriever.search() — dense + BM25 + RRF top-k.

Phase 7.8 — 그래프 통합 평가 확장 (DESIGN_GRAPHRAG §8):
  4) 평면 RRF (2신호): dense + BM25 만. graph_weight=0, summary_weight=0으로 토글.
  5) 통합 RRF (4신호): dense + BM25 + graph_proximity + community_summary (현 기본).
  같은 골드셋(기존 18 + 신규 graph 문항)에서 (4) vs (5)를 Hit@5·MRR로 비교한다.
  카테고리별로도 분리 집계해 기존 문항 회귀(Hit@5 하락) 여부와 graph 문항 수혜를 본다.

지표:
  - Hit@5 : 정답 노트가 top-5에 포함된 질문 비율. 모든 방식에 적용.
  - MRR   : 정답 노트의 역순위 평균(1/rank). 순위가 있는 방식에만 적용.
            grep은 무순위라 N/A (DESIGN §7 공정성 주의).

정직성 (절대):
  골드셋은 통계적 벤치마크가 아니라 방향성 지표다(표본 작음). 결과는 유불리와
  무관하게 그대로 보고한다. 그래프가 검색을 개선하지 못하면 "개선 못함 + 능력만 추가"로
  그대로 적는다. 골드셋을 통합 유리하게 조작하지 않는다 (DESIGN_GRAPHRAG §8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mycelium.core.config import Config
from mycelium.pipeline.retrieval import HybridRetriever

# 평가 상수
_TOP_K = 5  # Hit@k의 k

# 골드셋 파일 경로 (이 모듈과 같은 디렉토리)
_GOLDSET_PATH = Path(__file__).with_name("goldset.yaml")

# grep 베이스라인이 훑을 볼트 디렉토리 제외 목록 (인덱싱과 동일 기준 유지)
_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".obsidian", ".omc", ".git", ".venv", "__pycache__", "00-INBOX"}
)


# ---------------------------------------------------------------------------
# 데이터 구조
# ---------------------------------------------------------------------------


@dataclass
class GoldQuestion:
    """골드셋 질문 1건."""

    question: str
    expected_sources: list[str]
    category: str = ""


@dataclass
class MethodResult:
    """한 방식의 한 질문에 대한 평가 결과."""

    rank: int | None  # 정답 노트가 등장한 순위(1-based). 미발견이면 None.
    hit: bool  # top-k 안에 정답이 있었나


@dataclass
class MethodSummary:
    """한 방식의 전체 골드셋 집계."""

    name: str
    supports_mrr: bool  # MRR 정의 가능 여부 (grep=False)
    per_question: list[MethodResult] = field(default_factory=list)

    @property
    def hit_at_k(self) -> float:
        """Hit@k = 정답이 top-k에 든 질문 비율."""
        if not self.per_question:
            return 0.0
        hits = sum(1 for r in self.per_question if r.hit)
        return hits / len(self.per_question)

    @property
    def mrr(self) -> float | None:
        """MRR = 정답 역순위 평균. 미지원 방식(grep)은 None."""
        if not self.supports_mrr:
            return None
        if not self.per_question:
            return 0.0
        total = 0.0
        for r in self.per_question:
            if r.rank is not None and r.rank > 0:
                total += 1.0 / r.rank
        return total / len(self.per_question)


def _hit_at_k(results: list[MethodResult]) -> float:
    """주어진 MethodResult 부분집합의 Hit@k (카테고리별 집계용)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.hit) / len(results)


def _mrr(results: list[MethodResult]) -> float:
    """주어진 MethodResult 부분집합의 MRR (카테고리별 집계용)."""
    if not results:
        return 0.0
    total = 0.0
    for r in results:
        if r.rank is not None and r.rank > 0:
            total += 1.0 / r.rank
    return total / len(results)


# ---------------------------------------------------------------------------
# 골드셋 로딩
# ---------------------------------------------------------------------------


def load_goldset(path: Path | None = None) -> list[GoldQuestion]:
    """goldset.yaml을 읽어 GoldQuestion 리스트로 반환한다."""
    p = path or _GOLDSET_PATH
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    questions: list[GoldQuestion] = []
    for item in data.get("questions", []):
        questions.append(
            GoldQuestion(
                question=item["question"],
                expected_sources=list(item["expected_sources"]),
                category=item.get("category", ""),
            )
        )
    return questions


# ---------------------------------------------------------------------------
# 순위 계산 공통 유틸
# ---------------------------------------------------------------------------


def _first_match_rank(ranked_sources: list[str], expected: list[str]) -> int | None:
    """
    순위 매겨진 source 리스트에서 정답(expected) 중 하나가 처음 등장하는 순위를 찾는다.
    1-based. 못 찾으면 None. 첫 매칭이 곧 최선(가장 작은) 순위이므로 즉시 반환한다.
    """
    expected_set = set(expected)
    for rank, src in enumerate(ranked_sources, start=1):
        if src in expected_set:
            return rank
    return None


def _evaluate_ranking(
    ranked_sources: list[str], expected: list[str], k: int
) -> MethodResult:
    """순위 리스트로부터 (정답 순위, top-k 적중 여부)를 계산한다."""
    rank = _first_match_rank(ranked_sources, expected)
    hit = rank is not None and rank <= k
    return MethodResult(rank=rank, hit=hit)


# ---------------------------------------------------------------------------
# 1) grep 베이스라인 (무순위 → 매칭 단어 수로 정렬, MRR N/A)
# ---------------------------------------------------------------------------


def _collect_vault_files(vault_path: Path) -> list[Path]:
    """볼트 .md 파일을 인덱싱과 동일한 제외 기준으로 수집한다."""
    files: list[Path] = []
    for item in vault_path.rglob("*.md"):
        if any(part in _EXCLUDE_DIRS for part in item.parts):
            continue
        files.append(item)
    return files


def grep_baseline_rank(
    question: str, vault_path: Path, vault_files: list[Path]
) -> list[str]:
    """
    grep 베이스라인 — 현 워크플로(키워드 매칭) 대표.
    질문을 공백 분리한 단어들로 각 .md 본문을 매칭, 파일별 '매칭된 고유 단어 수'
    내림차순으로 정렬해 source 리스트를 반환한다.

    한국어 조사가 붙은 채로 매칭하므로 약할 것이며(예: '회복'이 '회복에'와 불일치),
    이 약점이 의미·하이브리드와의 대비 포인트다(DESIGN §7).
    """
    # 질문을 공백 분리 + 소문자화 (형태소 분석 없이 — grep의 한계를 재현)
    query_words = [w for w in question.lower().split() if w]
    if not query_words:
        return []

    scored: list[tuple[str, int]] = []
    for f in vault_files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace").lower()
        except OSError:
            continue
        # 파일 본문에 등장한 질문 단어의 고유 개수 = 매칭 점수
        matched = sum(1 for w in set(query_words) if w in text)
        if matched > 0:
            source = str(f.relative_to(vault_path))
            scored.append((source, matched))

    # 매칭 단어 수 내림차순 정렬 (동점은 입력 순서 유지 → 무순위 성격 반영)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [src for src, _ in scored]


# ---------------------------------------------------------------------------
# 평가 실행
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    """평가 전체 리포트."""

    grep: MethodSummary
    dense: MethodSummary
    hybrid: MethodSummary
    questions: list[GoldQuestion]
    # 질문별 (dense 순위, hybrid 순위, grep 순위) 기록 — 요약 출력용
    rank_rows: list[tuple[str, int | None, int | None, int | None]] = field(
        default_factory=list
    )
    # Phase 7.8 — 평면(2신호) vs 통합(4신호) 그래프 비교.
    flat: MethodSummary | None = None  # dense+BM25 (graph/summary weight=0)
    integrated: MethodSummary | None = None  # dense+BM25+graph+summary (현 기본)
    # 질문별 (flat 순위, integrated 순위) — 평면 vs 통합 표용
    graph_rank_rows: list[tuple[str, str, int | None, int | None]] = field(
        default_factory=list
    )  # (category, question, flat_rank, integ_rank)


def _rank_via_retriever(
    retriever: HybridRetriever, question: str, expected: list[str]
) -> MethodResult:
    """retriever.search() 결과를 노트 단위로 환산해 (순위, top-k 적중)을 계산한다."""
    chunks = retriever.search(question, k=max(_TOP_K * 4, 20))
    ranked = _dedup_sources([c.source for c in chunks])
    return _evaluate_ranking(ranked, expected, _TOP_K)


def run_evaluation(config: Config | None = None) -> EvalReport:
    """
    골드셋으로 grep/의미/하이브리드 3방식 + 평면 vs 통합 그래프 비교를 평가한다.
    HybridRetriever는 1회만 생성해 모든 방식이 같은 인덱스를 공유한다.

    평면(2신호) vs 통합(4신호)은 같은 retriever에서 graph_weight/summary_weight를
    런타임 토글해 측정한다(BM25 인덱스 중복 구축 없음). 통합 측정 후 원래 값으로 복원한다.
    """
    cfg = config or Config()
    questions = load_goldset()

    # 하이브리드 검색기 1개 생성 (BM25 인덱스 1회 구축, 모든 방식이 재사용)
    retriever = HybridRetriever(config=cfg)

    # grep용 볼트 파일 수집 (1회)
    vault_files = _collect_vault_files(cfg.vault_path)

    grep_summary = MethodSummary(name="grep (키워드만)", supports_mrr=False)
    dense_summary = MethodSummary(name="의미검색 단독", supports_mrr=True)
    hybrid_summary = MethodSummary(name="하이브리드", supports_mrr=True)
    flat_summary = MethodSummary(name="평면 RRF(2신호)", supports_mrr=True)
    integ_summary = MethodSummary(name="통합 RRF(4신호)", supports_mrr=True)

    rank_rows: list[tuple[str, int | None, int | None, int | None]] = []
    graph_rank_rows: list[tuple[str, str, int | None, int | None]] = []

    for q in questions:
        # 1) grep
        grep_ranked = grep_baseline_rank(q.question, cfg.vault_path, vault_files)
        grep_res = _evaluate_ranking(grep_ranked, q.expected_sources, _TOP_K)
        grep_summary.per_question.append(grep_res)

        # 2) 의미검색 단독 (top-k 안에서만 순위 판정 → 충분히 넓게 검색)
        dense_chunks = retriever.dense_search(q.question, k=max(_TOP_K * 4, 20))
        dense_ranked = _dedup_sources([c.source for c in dense_chunks])
        dense_res = _evaluate_ranking(dense_ranked, q.expected_sources, _TOP_K)
        dense_summary.per_question.append(dense_res)

        # 3) 하이브리드 (= 현 기본 설정의 통합 4신호와 동일하지만, 별도 표시 유지)
        hybrid_res = _rank_via_retriever(retriever, q.question, q.expected_sources)
        hybrid_summary.per_question.append(hybrid_res)

        rank_rows.append(
            (q.question, dense_res.rank, hybrid_res.rank, grep_res.rank)
        )

    # --- Phase 7.8: 평면(2신호) vs 통합(4신호) 비교 ---
    # 통합: 현재 weight 그대로. 평면: graph/summary weight를 0으로 토글.
    integ_results = _sweep_eval(retriever, questions, graph_w=None, summary_w=None)
    flat_results = _sweep_eval(retriever, questions, graph_w=0.0, summary_w=0.0)
    for q, fr, ir in zip(questions, flat_results, integ_results):
        flat_summary.per_question.append(fr)
        integ_summary.per_question.append(ir)
        graph_rank_rows.append((q.category, q.question, fr.rank, ir.rank))

    return EvalReport(
        grep=grep_summary,
        dense=dense_summary,
        hybrid=hybrid_summary,
        questions=questions,
        rank_rows=rank_rows,
        flat=flat_summary,
        integrated=integ_summary,
        graph_rank_rows=graph_rank_rows,
    )


def _sweep_eval(
    retriever: HybridRetriever,
    questions: list[GoldQuestion],
    graph_w: float | None,
    summary_w: float | None,
) -> list[MethodResult]:
    """
    주어진 graph_weight/summary_weight로 골드셋 전체를 평가해 MethodResult 리스트 반환.

    graph_w/summary_w가 None이면 현재(기본) 가중치를 그대로 사용한다.
    값을 주면 retriever 인스턴스 속성을 임시 교체해 측정 후 원래 값으로 복원한다
    (BM25 인덱스 재구축 없이 가중치만 토글 — 스윕·평면비교 공용 헬퍼).
    """
    # 원래 가중치 백업 (finally에서 반드시 복원).
    orig_g = retriever._graph_weight
    orig_s = retriever._summary_weight
    try:
        if graph_w is not None:
            retriever._graph_weight = graph_w
        if summary_w is not None:
            retriever._summary_weight = summary_w
        out: list[MethodResult] = []
        for q in questions:
            out.append(
                _rank_via_retriever(retriever, q.question, q.expected_sources)
            )
        return out
    finally:
        retriever._graph_weight = orig_g
        retriever._summary_weight = orig_s


@dataclass
class SweepRow:
    """가중치 스윕 1행 — (graph_w, summary_w)에서의 카테고리별 지표."""

    graph_w: float
    summary_w: float
    base_hit: float  # 기존 18문항 Hit@5 (회귀 감시 대상)
    base_mrr: float  # 기존 18문항 MRR
    graph_hit: float  # 신규 graph 문항 Hit@5
    graph_mrr: float  # 신규 graph 문항 MRR
    all_hit: float  # 전체 Hit@5
    all_mrr: float  # 전체 MRR


# 기존(비-graph) 카테고리 집합 — 회귀 감시 기준(기존 18문항).
_BASE_CATEGORIES: frozenset[str] = frozenset({"keyword", "paraphrase", "mixed"})


def run_weight_sweep(
    config: Config | None = None,
    graph_values: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0),
    summary_values: tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 1.0),
) -> tuple[list[SweepRow], list[GoldQuestion]]:
    """
    graph_weight·summary_weight를 스윕(D-3 방식)하며 골드셋을 평가한다.

    과적합 경계(DESIGN_GRAPHRAG §8): 기존 18문항 Hit@5를 떨어뜨리지 않으면서
    (회귀 0) graph 문항에서 이득이 있는 보수적 값을 고르기 위한 근거 데이터.

    스윕 차원을 곱집합으로 다 돌리면 25행이라 표가 길어진다. 가독성을 위해:
      - graph_weight 스윕(summary_weight=0 고정) + summary_weight 스윕(graph_weight=0 고정)
      의 두 1차원 스윕만 측정한다(상호작용은 작고 표본이 작아 곱집합은 과해석 위험).
    """
    cfg = config or Config()
    questions = load_goldset()
    retriever = HybridRetriever(config=cfg)

    rows: list[SweepRow] = []

    def _measure(gw: float, sw: float) -> SweepRow:
        results = _sweep_eval(retriever, questions, graph_w=gw, summary_w=sw)
        base = [r for r, q in zip(results, questions) if q.category in _BASE_CATEGORIES]
        graph = [r for r, q in zip(results, questions) if q.category == "graph"]
        return SweepRow(
            graph_w=gw,
            summary_w=sw,
            base_hit=_hit_at_k(base),
            base_mrr=_mrr(base),
            graph_hit=_hit_at_k(graph),
            graph_mrr=_mrr(graph),
            all_hit=_hit_at_k(results),
            all_mrr=_mrr(results),
        )

    # graph_weight 스윕 (summary_weight=0 고정)
    for gw in graph_values:
        rows.append(_measure(gw, 0.0))
    # summary_weight 스윕 (graph_weight=0 고정)
    for sw in summary_values:
        rows.append(_measure(0.0, sw))

    return rows, questions


def _dedup_sources(sources: list[str]) -> list[str]:
    """
    청크 단위 source 리스트를 노트 단위로 중복 제거한다(첫 등장 순위 유지).
    검색 결과는 청크 단위라 같은 노트가 여러 번 나올 수 있으므로 노트 순위로 환산.
    """
    seen: set[str] = set()
    deduped: list[str] = []
    for src in sources:
        if src not in seen:
            seen.add(src)
            deduped.append(src)
    return deduped


# ---------------------------------------------------------------------------
# 출력 포맷
# ---------------------------------------------------------------------------


def format_report(report: EvalReport) -> str:
    """EvalReport를 사람이 읽는 비교표 문자열로 포맷한다."""
    lines: list[str] = []
    n = len(report.questions)

    lines.append("=" * 64)
    lines.append(f"검색 품질 평가 — grep vs 의미검색 vs 하이브리드 (골드셋 {n}문항)")
    lines.append("=" * 64)
    lines.append("")

    # 비교표
    def _fmt_mrr(s: MethodSummary) -> str:
        m = s.mrr
        return "N/A (무순위)" if m is None else f"{m:.3f}"

    header = f"| {'방식':<16} | {'Hit@5':>7} | {'MRR':>13} |"
    sep = f"|{'-' * 18}|{'-' * 9}|{'-' * 15}|"
    lines.append(header)
    lines.append(sep)
    for s in (report.grep, report.dense, report.hybrid):
        hit_pct = f"{s.hit_at_k * 100:.1f}%"
        lines.append(f"| {s.name:<16} | {hit_pct:>7} | {_fmt_mrr(s):>13} |")
    lines.append("")

    # 질문별 정답 순위 요약 (의미/하이브리드/grep 각 순위. None은 '-'(top-k 밖))
    lines.append("질문별 정답 노트 순위 (rank, '-'=미적중/top밖)")
    lines.append("-" * 64)
    lines.append(f"{'#':>2}  {'의미':>4} {'HYB':>4} {'grep':>4}  카테고리   질문")
    for i, (q, d_rank, h_rank, g_rank) in enumerate(report.rank_rows, start=1):
        cat = report.questions[i - 1].category
        d = str(d_rank) if d_rank is not None else "-"
        h = str(h_rank) if h_rank is not None else "-"
        g = str(g_rank) if g_rank is not None else "-"
        q_short = q if len(q) <= 34 else q[:33] + "…"
        lines.append(f"{i:>2}  {d:>4} {h:>4} {g:>4}  {cat:<9} {q_short}")
    lines.append("")

    # --- Phase 7.8: 평면(2신호) vs 통합(4신호) 그래프 비교 ---
    if report.flat is not None and report.integrated is not None:
        lines.append("")
        lines.append("=" * 64)
        lines.append("평면 RRF(2신호) vs 통합 RRF(4신호) — 그래프 신호 on/off 비교")
        lines.append("=" * 64)
        lines.append("")
        lines.extend(_format_flat_vs_integrated(report))

    # 정직성 주석 (DESIGN §7)
    lines.append("-" * 64)
    lines.append(
        f"※ 표본 {n}문항은 통계적 벤치마크가 아닌 방향성 지표다. "
        "grep은 무순위라 MRR 미정의."
    )

    return "\n".join(lines)


def _format_flat_vs_integrated(report: EvalReport) -> list[str]:
    """평면(2신호) vs 통합(4신호) 비교표를 포맷한다 (Phase 7.8)."""
    assert report.flat is not None and report.integrated is not None
    lines: list[str] = []

    # 카테고리별 인덱스 분리 (기존 18문항 vs 신규 graph 문항).
    base_idx = [
        i for i, q in enumerate(report.questions) if q.category in _BASE_CATEGORIES
    ]
    graph_idx = [
        i for i, q in enumerate(report.questions) if q.category == "graph"
    ]

    def _subset(summary: MethodSummary, idx: list[int]) -> list[MethodResult]:
        return [summary.per_question[i] for i in idx]

    # 전체 / 기존 / graph 3구간 × 평면/통합 2방식 Hit@5·MRR 표.
    header = (
        f"| {'구간':<14} | {'평면 Hit@5':>10} {'통합 Hit@5':>10} "
        f"| {'평면 MRR':>9} {'통합 MRR':>9} |"
    )
    sep = f"|{'-' * 16}|{'-' * 23}|{'-' * 21}|"
    lines.append(header)
    lines.append(sep)

    segments = [
        (f"전체 ({len(report.questions)})", list(range(len(report.questions)))),
        (f"기존 ({len(base_idx)})", base_idx),
        (f"graph ({len(graph_idx)})", graph_idx),
    ]
    for label, idx in segments:
        if not idx:
            continue
        f_sub = _subset(report.flat, idx)
        i_sub = _subset(report.integrated, idx)
        lines.append(
            f"| {label:<14} | {_hit_at_k(f_sub) * 100:>9.1f}% "
            f"{_hit_at_k(i_sub) * 100:>9.1f}% "
            f"| {_mrr(f_sub):>9.3f} {_mrr(i_sub):>9.3f} |"
        )
    lines.append("")

    # 기존 18문항 회귀 판정 (Hit@5 하락 여부).
    f_base = _subset(report.flat, base_idx)
    i_base = _subset(report.integrated, base_idx)
    base_regress = _hit_at_k(i_base) < _hit_at_k(f_base)
    lines.append(
        f"기존 18문항 회귀 검사: 평면 Hit@5 {_hit_at_k(f_base) * 100:.1f}% → "
        f"통합 {_hit_at_k(i_base) * 100:.1f}%  "
        f"→ {'⚠️ 회귀 발생' if base_regress else '회귀 0 (Hit@5 유지)'}"
    )
    lines.append("")

    # 질문별 평면/통합 순위 (그래프가 순위를 올렸나 내렸나 그대로 표시).
    lines.append("질문별 평면 vs 통합 순위 (Δ = 통합-평면, 음수=통합이 더 높은 순위)")
    lines.append("-" * 64)
    lines.append(f"{'#':>2}  {'cat':<10} {'평면':>4} {'통합':>4} {'Δ':>4}  질문")
    for i, (cat, q, f_rank, i_rank) in enumerate(report.graph_rank_rows, start=1):
        f_s = str(f_rank) if f_rank is not None else "-"
        i_s = str(i_rank) if i_rank is not None else "-"
        if f_rank is not None and i_rank is not None:
            d = i_rank - f_rank
            d_s = f"{d:+d}" if d != 0 else "0"
        else:
            d_s = "?"
        q_short = q if len(q) <= 30 else q[:29] + "…"
        lines.append(f"{i:>2}  {cat:<10} {f_s:>4} {i_s:>4} {d_s:>4}  {q_short}")
    lines.append("")
    return lines


def format_weight_sweep(rows: list[SweepRow], questions: list[GoldQuestion]) -> str:
    """가중치 스윕 결과를 비교표로 포맷한다 (Phase 7.8 데이터튜닝 근거)."""
    n_base = sum(1 for q in questions if q.category in _BASE_CATEGORIES)
    n_graph = sum(1 for q in questions if q.category == "graph")

    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("가중치 데이터튜닝 스윕 (D-3) — graph_weight / summary_weight")
    lines.append("=" * 72)
    lines.append(
        f"기존 {n_base}문항(회귀 감시) | graph {n_graph}문항(능력 측정). "
        "기준: 기존 Hit@5 회귀 0 + graph 이득."
    )
    lines.append("")

    header = (
        f"| {'graph_w':>7} {'sum_w':>6} "
        f"| {'기존Hit':>7} {'기존MRR':>7} "
        f"| {'grHit':>6} {'grMRR':>6} "
        f"| {'전체Hit':>7} {'전체MRR':>7} |"
    )
    sep = f"|{'-' * 16}|{'-' * 17}|{'-' * 15}|{'-' * 17}|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r.graph_w:>7.1f} {r.summary_w:>6.1f} "
            f"| {r.base_hit * 100:>6.1f}% {r.base_mrr:>7.3f} "
            f"| {r.graph_hit * 100:>5.1f}% {r.graph_mrr:>6.3f} "
            f"| {r.all_hit * 100:>6.1f}% {r.all_mrr:>7.3f} |"
        )
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"※ 표본 한계: graph 문항 {n_graph}개는 방향성 지표일 뿐 통계적 유의성 없음. "
        "그래프가 검색을 못 키우면 노이즈 최소(낮은 weight) + '능력 추가'로 정직 보고."
    )
    return "\n".join(lines)


def main(config: Config | None = None) -> None:
    """
    CLI 진입점에서 호출 — 평가를 실행하고 비교표를 출력한다.

    출력 구성 (Phase 7.8):
      1) grep/의미/하이브리드 3방식 비교 + 질문별 순위 (Phase 4 기존)
      2) 평면(2신호) vs 통합(4신호) 그래프 비교 + 기존 18문항 회귀 검사
      3) 가중치 데이터튜닝 스윕 (graph_weight/summary_weight)
    """
    report = run_evaluation(config)
    print(format_report(report))
    print("")
    rows, questions = run_weight_sweep(config)
    print(format_weight_sweep(rows, questions))
