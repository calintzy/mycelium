"""
설정 모듈 — 볼트 경로, Chroma 경로, 임베딩 모델, 청킹 파라미터.
환경변수로 override 가능.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class Config:
    # 볼트 기본 경로 — 동봉된 합성 코퍼스 sample_vault/ (클론 즉시 동작).
    # 자신의 노트 폴더로 바꾸려면 VAULT_PATH 환경변수로 override 한다
    # (예: VAULT_PATH=~/MyNotes python -m mycelium index).
    # 기본값은 <프로젝트루트>/sample_vault — config.py 기준 parents[3]가 프로젝트 루트다.
    vault_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "VAULT_PATH",
                str(Path(__file__).parents[3] / "sample_vault"),
            )
        )
    )

    # Chroma 영속 경로 (환경변수 CHROMA_PATH로 override 가능)
    chroma_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "CHROMA_PATH",
                str(Path(__file__).parents[3] / "chroma"),
            )
        )
    )

    # 임베딩 모델명 (환경변수 EMBEDDING_MODEL로 override 가능)
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("EMBEDDING_MODEL", "bge-m3")
    )

    # Ollama 기본 URL (환경변수 OLLAMA_BASE_URL로 override 가능)
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://127.0.0.1:11434"
        )
    )

    # 2차 분할 임계 (자) — 이 길이 초과 섹션은 RecursiveCharacterTextSplitter로 재분할
    chunk_size: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_SIZE", "1000"))
    )

    # 2차 분할 overlap (자)
    chunk_overlap: int = field(
        default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP", "100"))
    )

    # 생성 LLM 모델명 (환경변수 GENERATION_MODEL로 override 가능)
    generation_model: str = field(
        default_factory=lambda: os.environ.get("GENERATION_MODEL", "qwen2.5:14b")
    )

    # 하이브리드 검색 RRF 가중치 (D-3, DESIGN §7 평가셋으로 결정)
    # dense=1.0 고정, bm25_weight=0.7 — 골드셋 스윕 결과 채택:
    #   0.7에서 전체 MRR 0.880(의미 단독 0.861 대비 +0.019),
    #   keyword MRR 0.833(동일가중 1.0과 동률, 손해 없음).
    #   0.5 이하에선 keyword MRR이 0.750으로 하락 → 키워드형 검색 품질 저하.
    #   단순하고 원칙("dense 주 신호, BM25 정밀 부스터")에 부합하는 값.
    dense_weight: float = field(
        default_factory=lambda: float(os.environ.get("DENSE_WEIGHT", "1.0"))
    )
    bm25_weight: float = field(
        default_factory=lambda: float(os.environ.get("BM25_WEIGHT", "0.7"))
    )

    # no_evidence 결정론 게이트 임계 (환경변수 RELEVANCE_THRESHOLD로 override 가능).
    # 최상위 dense 유사도(= 1 - cosine거리)가 이 값 미만이면 LLM 호출 전에
    # no_evidence=True로 확정한다(generation.py). LLM 출력 문구 판정의 취약성을 보완.
    #
    # 임계값 0.48 — 골드셋 18문항 측정 근거(bge-m3 임베딩, corpus 1046청크):
    #   정답 있는 질문 최상위 dense 유사도: 최소 0.5061 (장사 수익모델, paraphrase),
    #                                       최대 0.7745. 18문항 모두 0.50 이상.
    #   무근거 질의 최상위 dense 유사도: '오늘 날씨' 0.4554, '운동 후 단백질' 0.4424,
    #                                    '내일 점심' 0.4534, '주식 시장 전망' 0.4744. 모두 0.48 미만.
    #   → 0.48은 골드셋 최소(0.5061)보다 0.026 아래(정답 오거부 0 보장, 과적합 방지 마진),
    #     무근거 최대(0.4744)보다 0.006 위(무근거 질의 4건 전부 차단).
    #   정답이 1건이라도 오거부되면 이 값을 낮춰야 한다.
    relevance_threshold: float = field(
        default_factory=lambda: float(os.environ.get("RELEVANCE_THRESHOLD", "0.48"))
    )

    # Chroma 컬렉션명
    collection_name: str = "vault"

    # 그래프 근접 신호 RRF 가중치 (Phase 7.4, D-13. Phase 7.8 데이터튜닝으로 0.7→0.0 확정).
    # dense+BM25로 확보한 시드 노트의 그래프 이웃(N-홉) 노트의 대표청크에 부여하는 근접 순위를
    # RRF 4번째 신호로 융합할 때의 가중치.
    #
    # 7.8 스윕 근거(골드셋 기존18+graph4, graph_weight∈{0,0.3,0.5,0.7,1.0} 스윕, summary_w=0 고정):
    #   graph_weight | 기존MRR | graph문항MRR | graph Hit@5
    #          0.0   |  0.880  |  0.667       |  100%   ← 채택 (모든 지표 최고)
    #          0.3   |  0.880  |  0.646       |  100%   (graph 1문항이 1칸 밀림)
    #          0.5   |  0.852  |  0.383       |  100%
    #          0.7   |  0.801  |  0.362       |  100%   ← 종전 기본값(기존·graph 모두 하락)
    #          1.0   |  0.763  |  0.354       |   75%   (graph Hit@5까지 하락)
    # 정직한 결론(억측 없음, DESIGN_GRAPHRAG §8): 이 코퍼스에선 v1 dense 검색이 이미 강해
    #   (180노트·기존 골드셋 Hit@5 100%) 그래프 근접 신호는 검색을 전혀 개선하지 못한다.
    #   어떤 양수 가중에서도 graph_weight를 올릴수록 고연결 허브 노트를 끌어올려 정답
    #   순위를 오히려 떨어뜨린다(노이즈). graph_weight=0.3조차 신규
    #   graph 문항 MRR을 0.667→0.646으로 미세 하락시킨다. 따라서 검색 지표가 최적이고 회귀가
    #   완전 0인 graph_weight=0.0을 채택한다.
    #   ⚠️ 이는 그래프 기능을 폐기하는 것이 아니다. 그래프는 "검색 향상"이 아니라 "능력 추가"
    #   다 — 멀티홉 근거 표시(RetrievedChunk.graph_rank), 커뮤니티 조망(--exhaustive),
    #   ask 답변의 관계·커뮤니티 근거가 그래프의 가치다. RRF 순위 가중만 0으로 둔다.
    #   더 연결 밀도가 높거나 v1 검색이 약한 코퍼스에선 양수 가중이 의미를 가질 수 있다(코퍼스
    #   의존). GRAPH_WEIGHT 환경변수로 언제든 켤 수 있게 토글은 유지한다.
    graph_weight: float = field(
        default_factory=lambda: float(os.environ.get("GRAPH_WEIGHT", "0.0"))
    )

    # 그래프 근접 확장 홉 수 (Phase 7.4). 시드 노트에서 몇 홉까지 이웃 노트를 모을지.
    # 1~2홉 권장 — 2홉 초과는 무관 노트가 섞여 노이즈. 기본 2(직접 이웃 + 그 이웃).
    graph_hops: int = field(
        default_factory=lambda: int(os.environ.get("GRAPH_HOPS", "2"))
    )

    # 그래프 근접 시드 노트 수 (Phase 7.4). dense+BM25 상위 몇 개 노트를 그래프 확장 시드로 쓸지.
    # 너무 크면 그래프 신호가 질의와 무관하게 퍼져 노이즈. 기본 5.
    graph_seed_notes: int = field(
        default_factory=lambda: int(os.environ.get("GRAPH_SEED_NOTES", "5"))
    )

    # 커뮤니티 요약 검색단위 RRF 가중치 (Phase 7.5, D-13. Phase 7.8 데이터튜닝으로 0.7→0.3 확정).
    # community_summary는 이미 Chroma에 임베딩 적재되어 dense 검색에 섞이지만,
    # 요약 단위의 RRF 기여를 청크와 독립적으로 제어하기 위한 가중치.
    #
    # 7.8 스윕 근거(summary_weight∈{0,0.3,0.5,0.7,1.0} 스윕, graph_weight=0 고정):
    #   summary_weight 0.0~0.7 구간은 기존MRR 0.880·graph MRR 0.667로 완전 동률 —
    #   이 골드셋 정답이 모두 일반 노트라 요약 단위가 top 후보 순위에 거의 진입하지 않아
    #   영향이 없다. 1.0에서만 기존 0.870으로 미세 하락(요약이 끼어들어 청크를 한 칸 밀어냄).
    # 결론: graph_weight와 달리 summary_weight는 0.3에서 검색 지표를 전혀 떨어뜨리지 않는다
    #   (0.0과 완전 동률). 따라서 0.3을 채택한다 — 주제·조망 질문에서 커뮤니티 요약 단위가
    #   결과 합집합에 포함될 여지는 남기되(능력 추가), 기존 사실 질의 순위는 교란하지 않는
    #   보수값. 단, 이 골드셋 정답이 모두 일반 노트라 "요약이 정말 주제 질문을 돕는지"는
    #   이 표본으로 측정 못 한다(표본 한계). 주제 질문 전용 평가는 후속 과제로 남긴다.
    summary_weight: float = field(
        default_factory=lambda: float(os.environ.get("SUMMARY_WEIGHT", "0.3"))
    )

    # 그래프 영속 디렉토리 (환경변수 GRAPH_PATH로 override 가능, 기본: <프로젝트>/graph).
    # networkx 그래프(gpickle)·커뮤니티 요약 캐시·LLM 추출 해시캐시를 담는다 (Phase 7).
    # 원문 파생물이라 .gitignore 대상 (프라이버시, DESIGN_GRAPHRAG §13).
    graph_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "GRAPH_PATH",
                str(Path(__file__).parents[3] / "graph"),
            )
        )
    )

    # distill 위키노트 저장 디렉토리 (Phase 7.7, D-15). 볼트 루트 기준 상대경로.
    # distill이 큐레이션 위키노트(type:wiki, public:false)를 이 하위에 쓴다.
    # 볼트 안에 두어야 이후 index/graph-build에 자연히 포함된다(컴파운딩).
    # 환경변수 WIKI_DIR로 override 가능. 기본 "04-WIKI".
    wiki_dir: str = field(
        default_factory=lambda: os.environ.get("WIKI_DIR", "04-WIKI")
    )

    # 생성 산출물 제외 패턴 (Graphify/자동생성 파일 — 그래프·인덱싱에서 제외).
    # 실측상 graph_report.md(deg58)가 노이즈 허브였음 → 백본/보강 모두에서 배제 (D-11).
    # 파일명(소문자) 정확 매칭 + .graphify 접두 디렉토리/파일 매칭에 쓴다.
    graph_exclude_files: frozenset[str] = field(
        default_factory=lambda: frozenset({"graph_report.md", "graph.json"})
    )
    graph_exclude_prefixes: tuple[str, ...] = (".graphify",)

    def __post_init__(self) -> None:
        # 문자열로 넘어온 경우 Path로 변환
        self.vault_path = Path(self.vault_path)
        self.chroma_path = Path(self.chroma_path)
        self.graph_path = Path(self.graph_path)
        # SSRF 방어 — Ollama는 완전 로컬 전제(D-1). localhost/127.0.0.1만 허용.
        host = urlparse(self.ollama_base_url).hostname
        if host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError(
                f"OLLAMA_BASE_URL은 로컬 호스트만 허용합니다 (완전 로컬 전제, D-1). "
                f"허용: localhost/127.0.0.1/::1, 입력: {self.ollama_base_url!r}"
            )
