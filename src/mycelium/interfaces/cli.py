"""
CLI 인터페이스 — typer 앱.
Phase 1: index 명령 (인덱싱)
Phase 2: search 명령 (하이브리드 검색)
Phase 3: ask 명령 (RAG 답변 생성 + 출처 인용)
Phase 5: serve 명령 (MCP 서버 기동)
"""

from pathlib import Path

import typer

from mycelium.core.config import Config
from mycelium.pipeline.ingestion import run_ingestion

app = typer.Typer(
    name="mycelium",
    help="마크다운 볼트 하이브리드 검색 + RAG Q&A 도구",
    no_args_is_help=True,
)


@app.command()
def index(
    vault: str = typer.Option(
        None,
        "--vault",
        "-v",
        help="볼트 경로 (기본: 동봉된 sample_vault/ 또는 VAULT_PATH 환경변수)",
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
    public_only: bool = typer.Option(
        False,
        "--public-only",
        help="공개 전용 모드 (D-7): frontmatter public:true 노트만 인덱싱. "
        "실볼트 인덱스와 섞이지 않게 --chroma로 별도 경로를 지정하라.",
    ),
) -> None:
    """볼트 마크다운을 청킹·임베딩해 Chroma에 인덱싱한다."""
    config = Config()

    # CLI 옵션으로 경로 override
    if vault:
        config.vault_path = Path(vault)
    if chroma:
        config.chroma_path = Path(chroma)

    typer.echo(f"인덱싱 시작: {config.vault_path}")
    typer.echo(f"Chroma 경로: {config.chroma_path}")
    typer.echo(f"임베딩 모델: {config.embedding_model}")
    if public_only:
        typer.echo("모드: 공개 전용 (public:true 노트만, D-7 deny-by-default)")
    typer.echo("")

    # 공개 인덱싱 발 밟기 가드(ingestion) — 기본 경로면 ValueError. traceback 대신 깔끔히 안내.
    try:
        result = run_ingestion(config, public_only=public_only)
    except ValueError as e:
        typer.echo(f"오류: {e}")
        raise typer.Exit(1) from None

    typer.echo(
        f"{result.file_count} files, {result.chunk_count} chunks indexed"
        f" -> {result.chroma_path}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="검색 질의"),
    k: int = typer.Option(
        5,
        "--k",
        "-k",
        help="반환할 결과 수 (기본: 5)",
        min=1,
        max=50,
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
) -> None:
    """하이브리드 검색(dense + BM25 + RRF)으로 볼트를 검색한다."""
    from mycelium.pipeline.retrieval import HybridRetriever

    config = Config()
    if chroma:
        config.chroma_path = Path(chroma)

    typer.echo(f'검색 질의: "{query}"')
    typer.echo(f"Chroma 경로: {config.chroma_path}")
    typer.echo("BM25 인덱스 구축 중...")
    typer.echo("")

    retriever = HybridRetriever(config=config)

    # 코퍼스가 비어 있으면(인덱싱 전) 친절 안내 후 종료 (H4)
    if not retriever.has_corpus():
        typer.echo("인덱스가 비어 있습니다. 먼저 `index`를 실행하세요.")
        raise typer.Exit(1)

    results = retriever.search(query, k=k)

    if not results:
        typer.echo("검색 결과가 없습니다.")
        raise typer.Exit(1)

    typer.echo(f"검색 결과 {len(results)}건 (하이브리드 RRF 융합 점수 내림차순)")
    typer.echo("=" * 70)

    for i, chunk in enumerate(results, start=1):
        # 순위 표시 (dense/bm25 각 순위 병기)
        rank_info_parts = []
        if chunk.dense_rank is not None:
            rank_info_parts.append(f"dense #{chunk.dense_rank}")
        if chunk.bm25_rank is not None:
            rank_info_parts.append(f"bm25 #{chunk.bm25_rank}")
        if chunk.graph_rank is not None:
            rank_info_parts.append(f"graph #{chunk.graph_rank}")
        if chunk.kind == "community_summary":
            rank_info_parts.append(f"커뮤니티요약 #{chunk.community_id}")
        rank_info = ", ".join(rank_info_parts) if rank_info_parts else "합집합 보완"

        typer.echo(f"[{i}] RRF={chunk.rrf_score:.5f}  ({rank_info})")
        typer.echo(f"    source : {chunk.source}")
        if chunk.header_path:
            typer.echo(f"    header : {chunk.header_path}")
        typer.echo(f"    preview: {chunk.text_preview}")
        typer.echo("-" * 70)


@app.command(name="eval")
def evaluate(
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
) -> None:
    """
    골드셋으로 검색 품질을 비교한다.

    Phase 4: grep/의미검색/하이브리드 3방식 비교.
    Phase 7.8: 평면(2신호) vs 통합(4신호) 그래프 비교 + 가중치 데이터튜닝 스윕.
    """
    from mycelium.eval.evaluate import main as run_eval

    config = Config()
    if chroma:
        config.chroma_path = Path(chroma)

    typer.echo(f"평가 시작 (볼트: {config.vault_path})")
    typer.echo("BM25 인덱스 구축 및 검색 실행 중 (3방식 + 평면/통합 + 가중치 스윕)...")
    typer.echo("")

    run_eval(config)


@app.command()
def ask(
    question: str = typer.Argument(..., help="질문"),
    k: int = typer.Option(
        5,
        "--k",
        "-k",
        help="검색할 청크 수 (기본: 5)",
        min=1,
        max=50,
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
    exhaustive: bool = typer.Option(
        False,
        "--exhaustive",
        help="전수 조망 파워모드 (Phase 7.6, D-17): 통합검색 대신 전 커뮤니티 요약을 "
        "map-reduce로 빠짐없이 종합한다. 느리다(요약 수만큼 LLM콜) — 전체 주제·조망 질문용.",
    ),
) -> None:
    """RAG 답변 생성 — 질문에 대해 볼트를 근거로 답변하고 출처 노트를 표시한다."""
    from mycelium.pipeline.generation import generate_answer
    from mycelium.pipeline.retrieval import HybridRetriever

    config = Config()
    if chroma:
        config.chroma_path = Path(chroma)

    # --exhaustive 파워모드 — 전 커뮤니티 map-reduce 별도 경로(통합 경로와 분리, 회귀 없음).
    if exhaustive:
        _ask_exhaustive(question, config)
        return

    typer.echo(f'질문: "{question}"')
    typer.echo(f"검색 청크 수: {k}")
    typer.echo(f"생성 모델: {config.generation_model}")
    typer.echo("BM25 인덱스 구축 및 답변 생성 중... (첫 호출 시 모델 로드로 20~30초 소요)")
    typer.echo("")

    # 코퍼스가 비어 있으면(인덱싱 전) LLM 호출 전에 친절 안내 후 종료 (H4).
    # retriever를 미리 만들어 generate_answer에 재사용(중복 구축 방지).
    retriever = HybridRetriever(config=config)
    if not retriever.has_corpus():
        typer.echo("인덱스가 비어 있습니다. 먼저 `index`를 실행하세요.")
        raise typer.Exit(1)

    result = generate_answer(question=question, k=k, config=config, retriever=retriever)

    # 답변 출력
    typer.echo("=" * 70)
    typer.echo(result.text)
    typer.echo("")

    # 근거 없음 판정 시 출처 섹션 생략 — 답변과 출처가 모순되지 않게
    if not result.no_evidence:
        typer.echo("근거 노트:")
        typer.echo("-" * 70)
        for ref in result.sources:
            typer.echo(f"  [{ref.rank}] RRF={ref.rrf_score:.5f}  {ref.source}")
        typer.echo("=" * 70)
    else:
        typer.echo("=" * 70)


def _ask_exhaustive(question: str, config: Config) -> None:
    """`ask --exhaustive` 전수 조망 출력 (Phase 7.6) — 전 커뮤니티 map-reduce."""
    from mycelium.pipeline.graph.exhaustive import exhaustive_overview

    typer.echo(f'질문: "{question}" (전수 조망 모드 --exhaustive)')
    typer.echo(f"생성 모델: {config.generation_model}")
    typer.echo(
        "전 커뮤니티 요약을 map-reduce로 종합 중... "
        "(요약 수만큼 LLM 호출 — 느립니다)"
    )
    typer.echo("")

    result = exhaustive_overview(question=question, config=config)

    # 그래프·요약 미빌드 시 친절 안내 후 종료.
    if result.no_graph:
        typer.echo(
            "그래프 커뮤니티 요약이 없습니다. 먼저 `graph-build`를 실행하세요."
        )
        raise typer.Exit(1)

    # map 통계 — 전체 커뮤니티 중 관련/무관/실패 수를 분리 표시(전수 조망 신뢰성 지표).
    relevant = len(result.partials)
    typer.echo("=" * 70)
    typer.echo(
        f"[map] 커뮤니티 {result.total_communities}개 중 관련 {relevant}개, "
        f"무관 {result.skipped_irrelevant}개, 실패 {result.failed}개"
    )
    typer.echo("=" * 70)
    typer.echo("")

    # reduce 종합 답변.
    typer.echo(result.final_answer)
    typer.echo("")

    # 근거 커뮤니티 목록(부분답변이 채택된 군집).
    if result.partials:
        typer.echo("근거 커뮤니티:")
        typer.echo("-" * 70)
        for p in result.partials:
            typer.echo(f"  [커뮤니티 {p.community_id}] 노트 {p.note_count}개")
        typer.echo("=" * 70)
    else:
        typer.echo("=" * 70)


@app.command(name="graph-build")
def graph_build(
    vault: str = typer.Option(
        None,
        "--vault",
        "-v",
        help="볼트 경로 (기본: 동봉된 sample_vault/ 또는 VAULT_PATH 환경변수)",
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
    graph: str = typer.Option(
        None,
        "--graph",
        "-g",
        help="그래프 영속 경로 (기본: <프로젝트>/graph 또는 GRAPH_PATH 환경변수)",
    ),
) -> None:
    """그래프 인덱싱 (Phase 7) — 백본(위키링크) + LLM 엔티티 보강 + Leiden 커뮤니티·요약.

    LLM을 노트 수만큼 호출하므로 수 분 소요(캐시 후 빨라짐). 진행 로그가 출력된다.
    """
    from mycelium.pipeline.graph.build import build_graph

    config = Config()
    if vault:
        config.vault_path = Path(vault)
    if chroma:
        config.chroma_path = Path(chroma)
    if graph:
        config.graph_path = Path(graph)

    typer.echo(f"그래프 빌드 시작: {config.vault_path}")
    typer.echo(f"그래프 경로: {config.graph_path}")
    typer.echo(f"생성 모델: {config.generation_model} (엔티티 추출·요약)")
    typer.echo("")

    result = build_graph(config)

    typer.echo("")
    typer.echo("=" * 70)
    typer.echo(
        f"{result.nodes} nodes, {result.edges} edges, "
        f"{result.communities} communities, {result.summaries} summaries"
    )
    typer.echo(
        f"modularity {result.modularity:.3f} | "
        f"고립 노트 {result.isolated_before} → {result.isolated_after} (보강 효과)"
    )
    typer.echo(f"-> {result.graph_path}")
    typer.echo("=" * 70)


@app.command(name="distill")
def distill_cmd(
    topic: str = typer.Argument(..., help="정제할 주제 또는 질문"),
    vault: str = typer.Option(
        None,
        "--vault",
        "-v",
        help="볼트 경로 (기본: 동봉된 sample_vault/ 또는 VAULT_PATH 환경변수)",
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
    k: int = typer.Option(
        8,
        "--k",
        "-k",
        help="통합검색 근거 청크 수 (기본: 8)",
        min=1,
        max=50,
    ),
    note_date: str = typer.Option(
        None,
        "--date",
        help="위키노트 frontmatter date 값. 'today'면 오늘 날짜. 생략 시 date 필드 없음.",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        help="실제 저장 + 증분 인덱싱. 기본은 --dry-run(미리보기만, 저장 안 함).",
    ),
) -> None:
    """distill (Phase 7.7) — 주제/질문을 큐레이션 위키노트로 정제해 컴파운딩한다.

    통합검색으로 근거를 모아 LLM이 위키노트(type:wiki, public:false)를 작성하고,
    기존 그래프 노드에서 고른 [[관련노트]] 링크를 단다. 기본은 미리보기(--dry-run),
    --write일 때만 wiki_dir에 저장하고 그 청크만 증분 인덱싱한다(전체 재인덱싱 안 함).
    """
    from mycelium.pipeline.distill import distill as run_distill

    config = Config()
    if vault:
        config.vault_path = Path(vault)
    if chroma:
        config.chroma_path = Path(chroma)

    mode = "저장(--write)" if write else "미리보기(--dry-run)"
    typer.echo(f'주제: "{topic}"')
    typer.echo(f"볼트: {config.vault_path} | wiki_dir: {config.wiki_dir}")
    typer.echo(f"모드: {mode}")
    typer.echo("통합검색 + 위키노트 작성 중...")
    typer.echo("")

    result = run_distill(
        topic=topic, config=config, k=k, note_date=note_date, write=write
    )

    if result.no_evidence:
        typer.echo("근거가 부족합니다(검색 결과 없음). 먼저 `index`를 실행했는지 확인하세요.")
        raise typer.Exit(1)

    # 미리보기 — 생성된 위키노트 전문 출력.
    typer.echo("=" * 70)
    typer.echo(f"[생성 위키노트 미리보기] 파일명: {result.filename}")
    typer.echo("-" * 70)
    typer.echo(result.note_text)
    typer.echo("-" * 70)

    # 근거·링크 요약.
    if result.evidence_sources:
        typer.echo("근거 노트:")
        for src in result.evidence_sources:
            typer.echo(f"  - {src}")
    if result.links:
        typer.echo(f"관련 노트 링크 ({len(result.links)}개): " + ", ".join(result.links))

    typer.echo("=" * 70)

    if not write:
        typer.echo("미리보기만 했습니다(저장 안 함). 저장하려면 --write를 붙이세요.")
        return

    # 저장 결과.
    if result.skipped_existing:
        typer.echo(
            f"동일 파일명이 이미 존재하여 덮어쓰지 않고 .new로 저장했습니다 "
            f"-> {result.target_path}"
        )
    else:
        typer.echo(f"저장됨 -> {result.target_path}")
    typer.echo(f"증분 인덱싱: 청크 {result.indexed_chunks}개 추가 (커뮤니티 요약 보존).")
    typer.echo(
        "그래프 갱신은 증분 미지원 — 새 노드·엣지 반영하려면 `graph-build` 재실행이 필요합니다."
    )
    typer.echo("=" * 70)


@app.command()
def serve() -> None:
    """MCP 서버를 stdio 트랜스포트로 기동한다 (Claude Code 연동용, Phase 5)."""
    from mycelium.interfaces.mcp_server import run_server

    run_server()


@app.command()
def agentic(
    question: str = typer.Argument(..., help="질문"),
    k: int = typer.Option(
        5,
        "--k",
        "-k",
        help="검색할 청크 수 (기본: 5)",
        min=1,
        max=50,
    ),
    max_iter: int = typer.Option(
        2,
        "--max-iter",
        help="최대 검색 반복 횟수 (무한루프 가드, 기본: 2)",
        min=1,
        max=5,
    ),
    chroma: str = typer.Option(
        None,
        "--chroma",
        "-c",
        help="Chroma 영속 경로 (기본: <프로젝트>/chroma 또는 CHROMA_PATH 환경변수)",
    ),
) -> None:
    """에이전트형 RAG (Phase 6) — 검색 부족 시 질의를 재작성·재검색한다."""
    from mycelium.pipeline.agentic import run_agentic

    config = Config()
    if chroma:
        config.chroma_path = Path(chroma)

    typer.echo(f'질문: "{question}"')
    typer.echo(f"검색 청크 수: {k} / 최대 반복: {max_iter}")
    typer.echo(f"생성 모델: {config.generation_model}")
    typer.echo("에이전트 실행 중... (검색→관련성판정→충분하면 답변, 부족하면 재작성·재검색)")
    typer.echo("")

    result = run_agentic(question=question, k=k, max_iterations=max_iter, config=config)

    # --- 에이전트 동작 추적 출력 (에이전트 흐름이 보이게) ---
    typer.echo("=" * 70)
    typer.echo("[에이전트 동작]")
    typer.echo(f"  검색 반복 횟수: {result.iterations}")
    if result.rewrites:
        typer.echo("  질의 재작성 이력:")
        typer.echo(f"    [원본] {question}")
        for i, rq in enumerate(result.rewrites, start=1):
            typer.echo(f"    [재작성 {i}] {rq}")
    else:
        typer.echo("  질의 재작성: 없음 (1회 검색으로 충분)")
    if result.max_iterations_hit:
        typer.echo("  최대 반복 도달 → 가드로 종료 후 답변 생성")
    typer.echo("=" * 70)
    typer.echo("")

    # --- 최종 답변 출력 ---
    typer.echo(result.answer.text)
    typer.echo("")

    if not result.answer.no_evidence:
        typer.echo("근거 노트:")
        typer.echo("-" * 70)
        for ref in result.answer.sources:
            typer.echo(f"  [{ref.rank}] RRF={ref.rrf_score:.5f}  {ref.source}")
        typer.echo("=" * 70)
    else:
        typer.echo("=" * 70)
