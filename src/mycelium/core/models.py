"""
도메인 모델 — 청크 메타데이터 및 검색 결과 구조 정의.
Chroma 메타데이터는 str/int/float/bool만 허용하므로 dataclass로 정의 후 직렬화한다.
"""

from dataclasses import dataclass, field


@dataclass
class ChunkMetadata:
    """
    청크 하나의 메타데이터.
    - chunk_id: 청크 안정적 고유 ID (source + header_path + chunk_index). dense/BM25 공통 키 (C2).
    - source: 볼트 루트 기준 상대 파일 경로 (예: "03-KNOWLEDGE/foo.md")
    - header_path: 마크다운 헤더 계층 경로 (예: "## 섹션 > ### 하위섹션")
    - date: frontmatter date 필드 (없으면 빈 문자열)
    - project: frontmatter project 필드 (없으면 빈 문자열)
    - type: frontmatter type 필드 (없으면 빈 문자열)
    - public: frontmatter public 필드 (True/False, 없으면 False)
    """

    source: str = ""
    header_path: str = ""
    date: str = ""
    project: str = ""
    type: str = ""
    public: bool = False
    chunk_id: str = ""

    def to_chroma_dict(self) -> dict:
        """Chroma 메타데이터 딕셔너리로 변환 (str/int/float/bool 타입만 포함)."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "header_path": self.header_path,
            "date": self.date,
            "project": self.project,
            "type": self.type,
            "public": self.public,
        }

    @classmethod
    def from_frontmatter(
        cls,
        source: str,
        header_path: str,
        frontmatter: dict,
        chunk_id: str = "",
    ) -> "ChunkMetadata":
        """frontmatter 딕셔너리에서 ChunkMetadata를 생성한다."""

        def _str(key: str) -> str:
            val = frontmatter.get(key, "")
            return str(val) if val is not None else ""

        def _bool(key: str) -> bool:
            val = frontmatter.get(key, False)
            if isinstance(val, bool):
                return val
            # "true"/"True" 문자열도 처리
            return str(val).lower() == "true"

        return cls(
            source=source,
            header_path=header_path,
            date=_str("date"),
            project=_str("project"),
            type=_str("type"),
            public=_bool("public"),
            chunk_id=chunk_id,
        )


@dataclass
class RetrievedChunk:
    """
    하이브리드 검색 결과 1건.

    - source: 볼트 루트 기준 상대 파일 경로
    - header_path: 마크다운 헤더 계층 경로
    - text: 청크 본문 전체 텍스트 (LLM 생성 컨텍스트용, H1)
    - text_preview: 청크 본문 앞 200자 미리보기 (화면 표시용)
    - rrf_score: Reciprocal Rank Fusion 융합 점수 (높을수록 관련성↑)
    - dense_rank: 의미검색 순위 (없으면 None)
    - bm25_rank: BM25 키워드 검색 순위 (없으면 None)
    - graph_rank: 그래프 근접 순위 (Phase 7.4, 시드 노트의 N-홉 이웃 대표청크에 부여, 없으면 None)
    - kind: 검색 단위 종류 — 일반 청크는 "" (빈 문자열), 커뮤니티 요약은 "community_summary" (Phase 7.5)
    - community_id: kind=community_summary일 때 커뮤니티 번호 (혼합 granularity 출처 표시용)
    """

    source: str
    header_path: str
    text: str
    text_preview: str
    rrf_score: float
    dense_rank: int | None = None
    bm25_rank: int | None = None
    graph_rank: int | None = None
    kind: str = ""
    community_id: int | None = None


@dataclass
class SourceRef:
    """
    RAG 답변의 근거 노트 출처 정보 1건.

    - source: 볼트 루트 기준 상대 파일 경로
    - rrf_score: 해당 노트에서 가장 높은 RRF 점수 (청크 단위 최대값)
    - rank: 검색 결과 내 순위 (1-based)
    """

    source: str
    rrf_score: float
    rank: int


@dataclass
class Answer:
    """
    RAG 답변 생성 결과.

    - text: LLM이 생성한 답변 텍스트
    - sources: 근거로 사용된 출처 노트 목록 (중복 source 제거, 점수 내림차순)
    - no_evidence: LLM이 "노트에 근거가 없습니다" 류로 답한 경우 True
    """

    text: str
    sources: list[SourceRef] = field(default_factory=list)
    no_evidence: bool = False


# ---------------------------------------------------------------------------
# 그래프 도메인 모델 (Phase 7, DESIGN_GRAPHRAG §4)
# ---------------------------------------------------------------------------

# 노드 종류 식별자 (networkx 노드 속성 "kind"에 저장).
NODE_KIND_NOTE = "note"  # NoteNode — 노트 1개
NODE_KIND_ENTITY = "entity"  # EntityNode — LLM 추출 엔티티

# 엣지 종류 식별자 (networkx 엣지 속성 "kind"에 저장, DESIGN_GRAPHRAG §4).
EDGE_LINKS_TO = "links_to"  # 위키링크 [[ ]]
EDGE_RELATED = "related"  # frontmatter related
EDGE_TAGGED = "tagged"  # 공유 tags
EDGE_MENTIONS = "mentions"  # 노트 → 엔티티
EDGE_RELATES_TO = "relates_to"  # 엔티티 ↔ 엔티티 (관계 라벨)
