"""
하이브리드 검색 파이프라인 — dense(의미검색) + BM25(키워드) + graph_proximity + 커뮤니티 요약 RRF 융합.

설계 결정:
  D-8: 한국어 BM25는 kiwipiepy 형태소 토크나이저 사용. 실패 시 공백 분리로 graceful fallback.
  D-9: langchain_community(sunset)/langchain_classic 미사용. rank_bm25 + RRF 직접 구현.
  D-13: 그래프 근접·커뮤니티 요약을 별도 모드 없이 기존 RRF에 신호로 통합 (Phase 7.4/7.5).
        그래프 미존재 시 graceful — 기존 dense+BM25 2신호로 동작(회귀 없음).

클래스 구조:
  HybridRetriever — BM25 인덱스를 한 번 구축하고 반복 검색에 재사용.
  검색 1회당 BM25 재구축 없이 .search() 호출만으로 처리.
  그래프(graph.gpickle)는 최초 검색 시 1회 lazy 로드해 멤버에 보관(반복 검색 재사용).
"""

from __future__ import annotations

import warnings
from typing import Optional

from rank_bm25 import BM25Okapi

from mycelium.adapters.embedding import create_embeddings
from mycelium.adapters.vectorstore import create_vectorstore
from mycelium.core.config import Config
from mycelium.core.models import NODE_KIND_NOTE, RetrievedChunk

# 커뮤니티 요약 메타데이터 kind 값 (summarize.py와 동일 — 검색단위 구분용, Phase 7.5).
_SUMMARY_KIND = "community_summary"

# ---------------------------------------------------------------------------
# 한국어 형태소 토크나이저 (D-8)
# ---------------------------------------------------------------------------

# kiwi 인스턴스 — 모듈 레벨 lazy 싱글턴 (최초 호출 시 1회 초기화)
_kiwi_instance: object | None = None
_kiwi_fallback_warned: bool = False  # 경고는 1회만 출력


def _get_kiwi() -> object | None:
    """
    kiwipiepy Kiwi 인스턴스를 반환한다.
    import/초기화 실패 시 None을 반환하고 경고를 1회 출력한다.
    """
    global _kiwi_instance, _kiwi_fallback_warned
    if _kiwi_instance is not None:
        return _kiwi_instance

    try:
        from kiwipiepy import Kiwi  # type: ignore[import]

        _kiwi_instance = Kiwi()
        return _kiwi_instance
    except Exception as e:
        if not _kiwi_fallback_warned:
            warnings.warn(
                f"[mycelium] kiwipiepy 초기화 실패 — 공백 토크나이저로 대체합니다. "
                f"원인: {e}",
                stacklevel=2,
            )
            _kiwi_fallback_warned = True
        return None


def _tokenize(text: str) -> list[str]:
    """
    텍스트를 형태소 토큰 리스트로 변환한다.
    kiwi 사용 가능 시 형태소 분리, 불가 시 공백 분리 fallback.
    한국어 BM25 매칭 품질에 직결되는 핵심 함수 (D-8 참조).
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        # 공백 분리 fallback
        return text.lower().split()

    try:
        # Kiwi.tokenize() → Token 리스트, form 속성이 표층형
        tokens = kiwi.tokenize(text)  # type: ignore[union-attr]
        return [tok.form for tok in tokens if tok.form.strip()]
    except Exception:
        # 토크나이징 중 예외 시 공백 분리로 조용히 대체
        return text.lower().split()


# ---------------------------------------------------------------------------
# HybridRetriever 클래스
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Chroma(dense) + BM25(sparse) 하이브리드 검색기.

    초기화 시 Chroma에서 전체 청크를 로드해 BM25 인덱스를 1회 구축한다.
    이후 .search()는 BM25 재구축 없이 즉시 실행된다.

    Parameters
    ----------
    config : Config
        프로젝트 설정 (chroma_path, embedding_model 등).
    rrf_k : int
        RRF 분모 상수. 기본 60 (논문 표준값). 값이 클수록 순위 차이 둔감.
    dense_weight : float
        RRF에서 dense 결과에 부여할 가중치.
    bm25_weight : float
        RRF에서 BM25 결과에 부여할 가중치.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        rrf_k: int = 60,
        dense_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        graph_weight: Optional[float] = None,
        summary_weight: Optional[float] = None,
    ) -> None:
        self._config = config or Config()
        self._rrf_k = rrf_k
        # 가중치 기본값은 Config에서 가져온다 (D-3, DESIGN §7 스윕으로 결정).
        # 호출측에서 명시적 값을 넘기면 그 값이 우선된다(평가 스윕 등).
        self._dense_weight = dense_weight if dense_weight is not None else self._config.dense_weight
        self._bm25_weight = bm25_weight if bm25_weight is not None else self._config.bm25_weight
        # 그래프 근접(7.4)·커뮤니티 요약(7.5) 가중치 (D-13). 외부 주입 우선, 없으면 Config.
        self._graph_weight = (
            graph_weight if graph_weight is not None else self._config.graph_weight
        )
        self._summary_weight = (
            summary_weight if summary_weight is not None else self._config.summary_weight
        )

        # 임베딩·벡터스토어 초기화
        embeddings = create_embeddings(self._config)
        self._vectorstore = create_vectorstore(self._config, embeddings)

        # BM25 인덱스 구축 (Chroma → source-of-truth)
        self._corpus_ids: list[str] = []
        self._corpus_texts: list[str] = []
        self._corpus_metas: list[dict] = []
        self._bm25: BM25Okapi | None = self._build_bm25_index()

        # --- 그래프 근접(7.4) 준비물: lazy 로드 멤버 + source→chunk 역색인 ---
        # 그래프는 최초 검색 시 1회 로드해 _graph에 보관(_graph_loaded로 시도 여부 추적).
        # 그래프 미존재면 _graph=None으로 두고 dense+BM25 2신호로 graceful 동작(D-13).
        self._graph = None  # networkx.Graph | None (lazy)
        self._graph_loaded = False  # 로드 시도 완료 플래그(미존재 시 매번 재시도 방지)
        # 노트 source(상대경로) → 해당 노트의 청크 후보 리스트 [(key, corpus_idx), ...].
        # 노트→청크 전파(7.4)에서 그래프 이웃 노트의 청크를 빠르게 찾기 위한 역색인(1회 구축).
        # 커뮤니티 요약(kind=community_summary)은 노트 단위가 아니므로 제외.
        self._source_to_chunks: dict[str, list[tuple[str, int]]] = self._build_source_index()

    def has_corpus(self) -> bool:
        """인덱스에 청크가 1개 이상 있으면 True (인덱싱 여부 확인용, H4)."""
        return bool(self._corpus_texts)

    def top_dense_similarity(self, query: str) -> float | None:
        """
        질의에 대한 최상위 dense 유사도(= 1 - cosine거리)를 반환한다 (no_evidence 게이트용).

        no_evidence 결정론 게이트(generation.py)의 1차 신호로 쓰인다.
        하이브리드 RRF 점수는 순위 기반이라 "절대적 관련성" 임계를 잡기 어려운 반면,
        dense 유사도는 0~1 의미 척도라 골드셋으로 임계를 튜닝할 수 있다.
        코퍼스가 비어 있으면 None을 반환한다.

        주의(7.5 회귀 방지): 거리게이트 임계는 일반 청크 유사도로 튜닝된 값(0.48)이므로,
        커뮤니티 요약(kind=community_summary)은 게이트 판정에서 제외한다. 요약이 무근거
        질의에서 우연히 top1이 되어 게이트를 통과시키는 회귀를 막는다.
        """
        if not self._corpus_texts:
            return None
        # 요약이 섞여 들어와도 일반 청크 기준 최상위 유사도를 얻기 위해 여유 있게 조회.
        chunks = self.dense_search(query, k=5)
        for c in chunks:
            if c.kind == _SUMMARY_KIND:
                continue
            # dense_search는 rrf_score 필드에 유사도(1 - cosine거리)를 담는다.
            return c.rrf_score
        return None

    def dense_search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """
        dense(의미검색) 단독 경로. BM25/RRF 없이 Chroma similarity_search만 사용한다.
        평가(evaluate.py)에서 "의미검색 단독" 비교군에 쓰기 위한 공개 메서드.
        이미 구축된 vectorstore를 재사용하므로 BM25 인덱스 중복 구축이 없다.

        Returns
        -------
        list[RetrievedChunk]
            cosine 거리 오름차순(= 유사도 내림차순) 정렬 결과.
            rrf_score에는 "클수록 유사" 불변식을 지키기 위해 유사도(1 - cosine거리)를 담는다.
            dense_rank만 채운다(bm25_rank=None).
        """
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        chunks: list[RetrievedChunk] = []
        for rank, (doc, distance) in enumerate(results, start=1):
            meta = doc.metadata
            # cosine 거리(낮을수록 유사) → 유사도(높을수록 유사)로 변환해 불변식 유지.
            similarity = 1.0 - float(distance)
            is_summary = meta.get("kind") == _SUMMARY_KIND
            chunks.append(
                RetrievedChunk(
                    source=meta.get("source", ""),
                    header_path=meta.get("header_path", ""),
                    text=doc.page_content,
                    text_preview=doc.page_content[:200].replace("\n", " "),
                    rrf_score=similarity,
                    dense_rank=rank,
                    bm25_rank=None,
                    kind=_SUMMARY_KIND if is_summary else "",
                )
            )
        return chunks

    def _build_bm25_index(self) -> BM25Okapi | None:
        """
        Chroma에서 전체 청크를 가져와 BM25Okapi 인덱스를 구축한다.
        Chroma를 source-of-truth로 사용 — 별도 파일 저장 없음 (중복 저장 금지).
        코퍼스가 비어 있으면(인덱싱 전) None을 반환해 BM25Okapi 크래시를 피한다 (H4).
        """
        # Chroma 내부 클라이언트로 전체 문서 조회
        # langchain_chroma Chroma 객체는 ._collection.get()으로 chromadb 컬렉션에 접근
        raw = self._vectorstore._collection.get(
            include=["documents", "metadatas"]
        )

        ids: list[str] = raw.get("ids", [])
        documents: list[str] = raw.get("documents", []) or []
        metadatas: list[dict] = raw.get("metadatas", []) or []

        self._corpus_ids = ids
        self._corpus_texts = documents
        self._corpus_metas = metadatas

        # 빈 코퍼스 방어 (H4) — BM25Okapi는 빈 코퍼스에서 ZeroDivisionError를 던진다.
        if not documents:
            return None

        # 각 청크 텍스트를 형태소 토큰 리스트로 변환 (D-8 kiwi 토크나이저)
        tokenized_corpus: list[list[str]] = [
            _tokenize(text) for text in documents
        ]

        return BM25Okapi(tokenized_corpus)

    def _build_source_index(self) -> dict[str, list[tuple[str, int]]]:
        """
        노트 source(상대경로) → 그 노트에 속한 청크 후보 [(chunk_key, corpus_idx), ...] 역색인.

        그래프 노트 노드 id가 곧 노트 source(상대경로)이므로, 그래프 이웃 노트의 청크를
        이 역색인으로 즉시 찾는다 (노트→청크 전파, 7.4). corpus 순서(=chunk_index)를 보존해
        대표청크 선정 시 "노트 앞부분 청크"를 우선하기 쉽게 한다.
        커뮤니티 요약(kind=community_summary)은 노트 단위가 아니라 제외.
        """
        index: dict[str, list[tuple[str, int]]] = {}
        # key→source 역맵도 함께 구축(BM25-only 시드의 source 역조회용, O(1)).
        self._key_to_source: dict[str, str] = {}
        for corpus_idx, meta in enumerate(self._corpus_metas):
            if meta.get("kind") == _SUMMARY_KIND:
                continue
            src = meta.get("source", "")
            if not src:
                continue
            key = _make_key(meta)
            index.setdefault(src, []).append((key, corpus_idx))
            self._key_to_source[key] = src
        return index

    def _load_graph_lazy(self):
        """
        그래프(graph.gpickle)를 1회 lazy 로드해 반환한다 (없으면 None).
        그래프 미존재·로드 실패 시 None을 반환하고, 이후 호출에서 재시도하지 않는다
        (그래프 없는 환경에서 매 검색마다 파일 stat 비용 회피, D-13 graceful).
        """
        if self._graph_loaded:
            return self._graph
        self._graph_loaded = True
        try:
            from mycelium.adapters.graph_store import graph_exists, load_graph

            if not graph_exists(self._config):
                self._graph = None
                return None
            self._graph = load_graph(self._config)
        except Exception as e:  # noqa: BLE001 — 그래프 로드 실패는 검색을 막지 않음
            warnings.warn(
                f"[mycelium] 그래프 로드 실패 — 그래프 신호 없이 진행합니다. 원인: {e}",
                stacklevel=2,
            )
            self._graph = None
        return self._graph

    def _graph_proximity_ranks(
        self, seed_sources: list[str]
    ) -> dict[str, int]:
        """
        시드 노트(seed_sources)의 N-홉 그래프 이웃 노트를 모아, 이웃 노트의 대표청크에
        근접 순위(1-based)를 부여해 {chunk_key: graph_rank}로 반환한다 (7.4 핵심).

        흐름:
          1. 시드 노트에서 BFS로 graph_hops 홉까지 노트 이웃 확장.
             - 엣지 weight(빈도)·kind(links_to/related > tagged/mentions)로 근접 점수 가중.
             - 엔티티 노드는 "노트↔노트를 잇는 경유지"로만 통과(엔티티 자체는 결과 아님):
               노트→엔티티→노트 경로로 공유개념 연결 노트를 포착한다(D-11 보강 효과 활용).
          2. 가까운(점수 높은) 이웃 노트 순으로 정렬.
          3. **노트→청크 전파 + 도배 방지**: 노트당 대표청크 1개(코퍼스 첫 청크)에만
             근접 순위를 부여한다. 한 노트가 여러 청크로 결과를 도배하지 않게 하고,
             RRF는 순위 기반이라 대표청크 1개로도 그래프 신호가 충분히 전달된다.

        시드/그래프가 없으면 빈 dict.
        """
        graph = self._load_graph_lazy()
        if graph is None or not seed_sources:
            return {}

        hops = max(1, self._config.graph_hops)

        # --- 1) BFS로 N-홉 노트 이웃 확장 (엔티티는 경유지로만 통과) ---
        # note_score: 이웃 노트 source → 누적 근접 점수(클수록 가까움). 시드 자신은 제외.
        note_score: dict[str, float] = {}
        seed_set = {s for s in seed_sources if graph.has_node(s)}
        if not seed_set:
            return {}

        # (현재 노트, 남은 홉, 누적 가중) 큐. 같은 노트를 더 가까운 경로로 재방문하면 갱신.
        visited_best: dict[str, float] = {s: float(hops + 1) for s in seed_set}
        frontier: list[tuple[str, int, float]] = [
            (s, hops, float(hops + 1)) for s in seed_set
        ]

        while frontier:
            node, remaining, acc = frontier.pop()
            if remaining <= 0:
                continue
            for nbr in graph.neighbors(node):
                edata = graph[node][nbr]
                nbr_kind = graph.nodes[nbr].get("kind")
                # 엣지 kind별 가중 — 명시적 연결(links_to/related)을 공유태그/언급보다 강하게.
                # 가중 스케일을 크게(0.5×) 두어 직접 노트 이웃(links_to)이 엔티티 경유 노트보다
                # 확실히 상위에 오게 한다(연결 노트가 RRF 컷오프 안에 들어오도록).
                ekind = edata.get("kind")
                if ekind in ("links_to", "related"):
                    edge_w = 1.0
                elif ekind in ("relates_to",):
                    edge_w = 0.6
                else:  # tagged / mentions — 약한 신호
                    edge_w = 0.4
                weight = float(edata.get("weight", 1.0))
                # 홉이 멀수록 감쇠(remaining이 작을수록 멀다) + 엣지 가중·빈도 반영.
                step_score = acc - 1.0 + 0.5 * edge_w * weight

                if nbr_kind == NODE_KIND_NOTE:
                    # 이웃 노트 — 근접 점수 누적(더 가까운 경로면 갱신).
                    if nbr not in seed_set:
                        prev = note_score.get(nbr, 0.0)
                        if step_score > prev:
                            note_score[nbr] = step_score
                    # 노트를 경유해 더 확장(남은 홉 -1).
                    if visited_best.get(nbr, -1.0) < step_score:
                        visited_best[nbr] = step_score
                        frontier.append((nbr, remaining - 1, step_score))
                else:
                    # 엔티티 노드 — 결과 노트는 아니지만 경유지로 통과(노트↔노트 공유개념 연결).
                    # 엔티티 2회 경유 ≈ 노트 1홉 예산: remaining을 0.5 차감(약한 홉예산).
                    # graph_weight=0이라 현재 RRF 순위에는 영향 없지만, 양수 전환 대비로
                    # 먼 노트까지 무제한 퍼지는 것을 방지한다(토글 안전성).
                    via_score = step_score - 1.0
                    via_remaining = remaining - 0.5
                    if visited_best.get(nbr, -1.0) < via_score:
                        visited_best[nbr] = via_score
                        frontier.append((nbr, via_remaining, via_score))

        if not note_score:
            return {}

        # --- 2) 근접 점수 내림차순 정렬 → 이웃 노트 순위 ---
        ranked_notes = sorted(note_score.items(), key=lambda x: x[1], reverse=True)

        # --- 3) 노트→대표청크 전파 (노트당 1청크, 도배 방지) ---
        proximity_ranks: dict[str, int] = {}
        rank = 0
        for src, _score in ranked_notes:
            chunks = self._source_to_chunks.get(src)
            if not chunks:
                continue  # 그래프엔 있으나 인덱싱 안 된 노트(생성산출물 등) — 스킵.
            # 대표청크 = 코퍼스 등장 순서상 첫 청크(보통 chunk_index 0, 노트 도입부).
            rep_key = chunks[0][0]
            if rep_key in proximity_ranks:
                continue
            rank += 1
            proximity_ranks[rep_key] = rank
        return proximity_ranks

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        하이브리드 검색을 실행하고 RetrievedChunk 리스트를 반환한다.

        Parameters
        ----------
        query : str
            검색 질의.
        k : int
            반환할 최종 결과 수.

        Returns
        -------
        list[RetrievedChunk]
            RRF 융합 점수 내림차순으로 정렬된 결과 리스트.
        """
        # 빈 코퍼스 방어 (H4) — 인덱싱 전이면 dense·BM25 모두 결과가 없으므로 빈 결과 반환.
        if not self._corpus_texts:
            return []

        # candidate 수는 k의 3배로 넉넉하게 (두 결과 합집합에서 최종 k개 선별)
        candidate_k = max(k * 3, 20)

        # 1) dense 검색 (Chroma 의미검색)
        dense_results = self._vectorstore.similarity_search_with_score(
            query, k=candidate_k
        )
        # dense_results: [(Document, distance), ...] — distance는 cosine 거리 (낮을수록 유사)

        # dense 결과를 chunk_id(키) → 순위(1-based) 매핑.
        # community_summary 단위(7.5)도 dense 검색에 섞여 나오므로 키 집합에 자연 포함된다.
        # 요약은 summary_set으로 표시해 RRF에서 summary_weight를 적용한다(청크와 독립 가중).
        dense_rank_map: dict[str, int] = {}
        dense_doc_map: dict[str, tuple] = {}  # 키 → (Document, distance)
        summary_set: set[str] = set()  # community_summary 단위 키 집합 (7.5)

        for rank, (doc, distance) in enumerate(dense_results, start=1):
            meta = doc.metadata
            key = _make_key(meta)
            dense_rank_map[key] = rank
            dense_doc_map[key] = (doc, distance)
            if meta.get("kind") == _SUMMARY_KIND:
                summary_set.add(key)

        # 2) BM25 키워드 검색 (코퍼스가 비어 _bm25=None이면 dense-only로 진행, H4)
        bm25_rank_map: dict[str, int] = {}
        bm25_idx_map: dict[str, int] = {}  # 키 → corpus 인덱스

        if self._bm25 is not None:
            query_tokens = _tokenize(query)
            bm25_scores: list[float] = self._bm25.get_scores(query_tokens)

            # BM25 scores를 내림차순 정렬 → 상위 candidate_k개 추출.
            # score>0 후보만 사용 — 매칭 토큰이 없는(0점) 청크는 BM25 신호가 아님 (M/L).
            indexed_scores = sorted(
                enumerate(bm25_scores), key=lambda x: x[1], reverse=True
            )[:candidate_k]

            rank = 0
            for corpus_idx, score in indexed_scores:
                if score <= 0:
                    continue
                if corpus_idx >= len(self._corpus_metas):
                    continue
                rank += 1
                meta = self._corpus_metas[corpus_idx]
                key = _make_key(meta)
                bm25_rank_map[key] = rank
                bm25_idx_map[key] = corpus_idx

        # 2.5) graph_proximity (7.4) — dense+BM25 융합으로 확보한 상위 노트를 시드로,
        #      그래프 이웃 노트의 대표청크에 근접 순위를 부여한다.
        #      시드 산출은 dense+BM25 2신호 RRF로 먼저 정렬한 노트(요약 제외)에서 추출한다
        #      — 그래프 신호 자체가 시드 선정에 끼어들지 않게(순환 방지) 2신호 기준으로 뽑는다.
        seed_sources = self._select_seed_sources(
            dense_rank_map, bm25_rank_map, dense_doc_map, summary_set
        )
        graph_rank_map = self._graph_proximity_ranks(seed_sources)

        # 3) RRF 융합 — dense + BM25 + graph_proximity + community_summary 합집합 (D-9, D-13)
        # weight=0인 신호는 0점이므로 합집합에서 미리 제외한다.
        # graph_weight=0 → 통합이 평면 dense+BM25와 정확히 동일 결과 보장.
        # summary_weight=0 → 커뮤니티 요약이 순위에 끼어들지 않음.
        all_keys = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
        if self._graph_weight > 0:
            all_keys |= set(graph_rank_map.keys())
        if self._summary_weight == 0:
            all_keys -= summary_set

        rrf_scores: dict[str, float] = {}
        for key in all_keys:
            score = 0.0
            if key in dense_rank_map:
                # 요약 단위는 summary_weight, 일반 청크는 dense_weight (7.5 독립 가중).
                w = self._summary_weight if key in summary_set else self._dense_weight
                score += w / (self._rrf_k + dense_rank_map[key])
            if key in bm25_rank_map:
                score += self._bm25_weight / (self._rrf_k + bm25_rank_map[key])
            if key in graph_rank_map:
                score += self._graph_weight / (self._rrf_k + graph_rank_map[key])
            rrf_scores[key] = score

        # 융합 점수 내림차순 정렬 → 상위 k개
        top_keys = sorted(rrf_scores.keys(), key=lambda k_: rrf_scores[k_], reverse=True)[:k]

        # graph_proximity 전용 키(dense/bm25 합집합에 없던 그래프 이웃 청크)의 텍스트 조회용
        # 역색인: chunk_key → corpus_idx (대표청크만 들어와 있어 소수).
        graph_only_idx: dict[str, int] = {}
        for src, chunk_list in self._source_to_chunks.items():
            for ckey, cidx in chunk_list:
                if ckey in graph_rank_map:
                    graph_only_idx[ckey] = cidx

        # 4) RetrievedChunk 조립
        results: list[RetrievedChunk] = []
        for key in top_keys:
            # 텍스트와 메타데이터 조회 (dense 우선, BM25 corpus, 마지막으로 graph-only corpus)
            if key in dense_doc_map:
                doc, _distance = dense_doc_map[key]
                text = doc.page_content
                meta = doc.metadata
            elif key in bm25_idx_map:
                corpus_idx = bm25_idx_map[key]
                text = self._corpus_texts[corpus_idx]
                meta = self._corpus_metas[corpus_idx]
            elif key in graph_only_idx:
                # 그래프 근접으로만 등장한 청크(dense/BM25 후보엔 없던 연결 노트) — corpus에서 조회.
                corpus_idx = graph_only_idx[key]
                text = self._corpus_texts[corpus_idx]
                meta = self._corpus_metas[corpus_idx]
            else:
                continue

            # community_summary 단위면 community_id를 함께 실어 출처 표시(혼합 granularity, 7.5).
            is_summary = meta.get("kind") == _SUMMARY_KIND
            community_id = meta.get("community_id") if is_summary else None
            results.append(
                RetrievedChunk(
                    source=meta.get("source", ""),
                    header_path=meta.get("header_path", ""),
                    text=text,  # 전체 본문 (LLM 컨텍스트용, H1)
                    text_preview=text[:200].replace("\n", " "),
                    rrf_score=rrf_scores[key],
                    dense_rank=dense_rank_map.get(key),
                    bm25_rank=bm25_rank_map.get(key),
                    graph_rank=graph_rank_map.get(key),
                    kind=_SUMMARY_KIND if is_summary else "",
                    community_id=int(community_id) if community_id is not None else None,
                )
            )

        return results

    def _select_seed_sources(
        self,
        dense_rank_map: dict[str, int],
        bm25_rank_map: dict[str, int],
        dense_doc_map: dict[str, tuple],
        summary_set: set[str],
    ) -> list[str]:
        """
        dense+BM25 2신호 RRF로 상위 청크를 정렬해, 그 청크들의 노트 source를
        그래프 확장 시드로 추출한다 (graph_seed_notes개, 중복 제거, 순위 보존).

        그래프 신호를 시드 선정에 넣지 않는 이유: 그래프 근접은 "이미 관련 있는 노트"에서
        퍼져 나가야 의미가 있다. 그래프로 뽑은 노트를 다시 시드로 쓰면 신호가 자기참조로
        번져 노이즈가 된다. 따라서 시드는 순수 dense+BM25(요약 제외) 기준으로 뽑는다.
        """
        n_seed = max(1, self._config.graph_seed_notes)
        seed_keys = set(dense_rank_map.keys()) | set(bm25_rank_map.keys())
        # 요약 단위는 노트가 아니므로 시드에서 제외.
        seed_keys -= summary_set

        # 2신호 RRF 점수로 정렬.
        seed_scores: dict[str, float] = {}
        for key in seed_keys:
            s = 0.0
            if key in dense_rank_map:
                s += self._dense_weight / (self._rrf_k + dense_rank_map[key])
            if key in bm25_rank_map:
                s += self._bm25_weight / (self._rrf_k + bm25_rank_map[key])
            seed_scores[key] = s

        ordered = sorted(seed_scores.keys(), key=lambda k_: seed_scores[k_], reverse=True)

        # 청크 → source 변환(노트 단위 중복 제거, 상위 순위 노트 우선).
        seeds: list[str] = []
        seen: set[str] = set()
        for key in ordered:
            # dense 후보면 doc 메타에서, 아니면 _make_key 역추적 대신 source_to_chunks 역참조.
            src = ""
            if key in dense_doc_map:
                src = dense_doc_map[key][0].metadata.get("source", "")
            if not src:
                # BM25-only 키 — 키 자체가 source::header::idx 형태라 청크 메타에서 조회.
                src = self._source_of_key(key)
            if not src or src in seen:
                continue
            seen.add(src)
            seeds.append(src)
            if len(seeds) >= n_seed:
                break
        return seeds

    def _source_of_key(self, key: str) -> str:
        """chunk_key로부터 노트 source를 역조회한다 (key→source 역맵, O(1))."""
        return self._key_to_source.get(key, "")


# ---------------------------------------------------------------------------
# 내부 유틸리티
# ---------------------------------------------------------------------------


def _make_key(meta: dict) -> str:
    """
    청크 안정적 고유 ID(chunk_id)를 dense/BM25 공통 키로 사용한다 (C2).
    ingestion에서 source+header_path+chunk_index로 부여한 값이라 sub-chunk·헤더없는 청크가
    각각 고유 키를 가진다. chunk_id가 없는 구버전 인덱스는 source::header_path로 폴백.
    """
    cid = meta.get("chunk_id", "")
    if cid:
        return cid
    return f"{meta.get('source', '')}::{meta.get('header_path', '')}"
