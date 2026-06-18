"""
KorQuAD v1 독립 벤치마크 — 하이브리드(+kiwi BM25)가 dense 단독을 실제로 이기는지 측정.

목적
----
우리 볼트(180노트)는 dense가 이미 Hit@5 100%라 검색 기법 우위가 보이지 않는다.
포화되지 않은 공개 한국어 벤치마크(KorQuAD v1)의 서브셋에서 세 가지 검색 방식을
같은 코퍼스로 비교해, kiwi 형태소 BM25 하이브리드의 실제 이득을 정직하게 측정한다.

비교 3방식 (모두 같은 코퍼스·같은 dense 인덱스 사용)
  1) dense 단독            — Chroma 의미검색만 (bge-m3)
  2) dense + 공백 BM25     — off-the-shelf 하이브리드, RRF 융합
  3) dense + kiwi BM25     — 우리 하이브리드(retrieval.py 방식), RRF 융합

지표: Hit@5, Hit@10, MRR@10 (단락=문서 단위 정답 매칭)

정직성 원칙
  - dense가 포화되면(Hit@5 100%) 코퍼스를 키워 난이도를 높여 포화를 깬다.
    그래도 안 깨지면 그 사실을 그대로 보고한다.
  - kiwi/하이브리드가 dense를 못 이기면 결과를 조작하지 않고 그대로 보고한다.
  - 서브셋·표본 규모를 결과 표에 명시한다.

격리 원칙 (제약)
  - 시스템 추론은 로컬 Ollama(bge-m3)만 사용 — 외부 LLM/임베딩 API 금지.
  - 데이터셋 다운로드는 "평가용 골드셋 확보"이며 시스템 추론(로컬)과 무관하다.
  - 볼트 인덱스(repo의 chroma/)·실볼트는 절대 미접촉. 별도 경로(/tmp)만 사용한다.
  - 기존 mycelium 코드는 읽기만 하고 수정하지 않는다(독립 스크립트).

재사용
  - 임베딩: mycelium.adapters.embedding.create_embeddings (OllamaEmbeddings, bge-m3)
  - kiwi 토크나이저: mycelium.pipeline.retrieval._tokenize (D-8 방식 동일)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

from rank_bm25 import BM25Okapi

# 기존 코드 재사용 (읽기 전용) — 임베딩 팩토리와 kiwi 토크나이저.
from mycelium.adapters.embedding import create_embeddings
from mycelium.core.config import Config
from mycelium.pipeline.retrieval import _tokenize as kiwi_tokenize

# ---------------------------------------------------------------------------
# 서브셋 규모 (결과에 명시). 환경변수로 조정 가능 — 포화 시 코퍼스↑로 난이도 상향.
# ---------------------------------------------------------------------------
# 코퍼스 단락 수: 클수록 dense가 "헷갈릴" 후보가 많아져 포화가 깨진다.
N_CONTEXTS = int(os.environ.get("KORQUAD_N_CONTEXTS", "2000"))
# 질의(질문) 수: 정답 매칭 안정성을 위해 충분히.
N_QUERIES = int(os.environ.get("KORQUAD_N_QUERIES", "500"))
# 평가 cutoff
K_LIST = (5, 10)
MRR_K = 10
RRF_K = 60  # retrieval.py와 동일 표준값
# RRF 가중치 — Config 기본값(dense=1.0, bm25=0.7)을 그대로 사용해 우리 시스템 설정 반영.
_cfg_for_weights = Config()
DENSE_WEIGHT = _cfg_for_weights.dense_weight
BM25_WEIGHT = _cfg_for_weights.bm25_weight

# 별도 Chroma 경로 — 볼트 인덱스와 완전 분리(/tmp). 매 실행 시 새로 구축.
KORQUAD_CHROMA_PATH = Path(os.environ.get("KORQUAD_CHROMA_PATH", "/tmp/korquad_chroma"))
KORQUAD_COLLECTION = "korquad_bench"

# KorQuAD v1 공식 dev JSON (datasets 실패 시 직접 다운로드용 fallback).
KORQUAD_V1_DEV_URL = (
    "https://raw.githubusercontent.com/korquad/korquad.github.io/master/dataset/KorQuAD_v1.0_dev.json"
)


# ---------------------------------------------------------------------------
# 1) 데이터셋 로드 — 고유 단락(context) 코퍼스 + 질문 질의 + gold 매핑
# ---------------------------------------------------------------------------
def _load_korquad_records() -> tuple[list[dict], list[str]]:
    """
    KorQuAD v1 데이터를 반환한다.

    반환
    ----
    query_records : list[dict]  — 질의용 (질문, 정답 context). validation 스플릿 기준.
                                   각 레코드 {"question": str, "context": str}.
    distractor_contexts : list[str] — 코퍼스 확대용 추가 단락 풀(train 스플릿의 고유 단락).
                                   포화 시 코퍼스를 키워 난이도를 높이는 distractor로만 쓰인다.
                                   (이 단락들이 정답인 질문은 질의에 넣지 않는다.)

    설계 메모
      validation 고유 단락은 ~960개뿐이라, 코퍼스를 그 이상으로 키워 dense 포화를
      깨려면 단락 풀이 부족하다. 따라서 train 스플릿의 단락을 distractor(오답 후보
      단락)로만 추가 투입해 코퍼스 난이도를 통제한다. 질의(질문)와 gold는 항상
      validation 기준이며 평가의 공정성을 유지한다.

    1순위: datasets 라이브러리('KorQuAD/squad_kor_v1').
    2순위(실패 시): 공식 dev JSON 직접 다운로드(이 경우 distractor 풀 없음).
    """
    # --- 1순위: datasets ---
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds_val = load_dataset("KorQuAD/squad_kor_v1", split="validation")
        query_records = [
            {"question": row["question"], "context": row["context"]}
            for row in ds_val
            if row.get("question") and row.get("context")
        ]
        # train 스플릿의 고유 단락을 distractor 풀로 (질문은 쓰지 않고 단락만).
        ds_train = load_dataset("KorQuAD/squad_kor_v1", split="train")
        distractors: list[str] = []
        seen_ctx: set[str] = set()
        for row in ds_train:
            ctx = row.get("context")
            if ctx and ctx not in seen_ctx:
                seen_ctx.add(ctx)
                distractors.append(ctx)
        if query_records:
            print(
                f"[데이터셋] datasets 'KorQuAD/squad_kor_v1' validation 로드: "
                f"{len(query_records)}개 QA / train distractor 단락 풀: {len(distractors)}개"
            )
            return query_records, distractors
    except Exception as e:
        print(f"[데이터셋] datasets 로드 실패 → JSON 직접 다운로드로 대체. 원인: {e}")

    # --- 2순위: 공식 JSON 직접 다운로드 ---
    print(f"[데이터셋] 공식 JSON 다운로드: {KORQUAD_V1_DEV_URL}")
    with urllib.request.urlopen(KORQUAD_V1_DEV_URL, timeout=60) as resp:
        raw = json.loads(resp.read().decode("utf-8"))
    records = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("question") and context:
                    records.append({"question": qa["question"], "context": context})
    print(f"[데이터셋] 공식 dev JSON 로드: {len(records)}개 QA (distractor 풀 없음)")
    return records, []  # JSON fallback은 distractor 풀 제공 불가.


def build_subset(records: list[dict], distractors: list[str], n_contexts: int, n_queries: int):
    """
    레코드+distractor 풀에서 서브셋을 구성한다 — 규모 통제를 위해.

    반환
    ----
    corpus_ids   : list[str]   — 단락 id (doc id, 정답 매칭 단위)
    corpus_texts : list[str]   — 고유 단락 본문 (dedup 완료)
    queries      : list[dict]  — [{"question": str, "gold_id": str}, ...]

    절차
      1) 질의는 먼저 확정한다 — validation 레코드에서 앞쪽 n_queries개 질문(중복 제거).
         각 질문의 정답 단락은 반드시 코퍼스에 포함시킨다(gold 보장).
      2) 코퍼스 = (질의들의 gold 단락) + (나머지 validation 고유 단락) + (train distractor 단락).
         앞에서부터 n_contexts개가 될 때까지 채운다. gold 단락은 무조건 포함되어
         "정답이 코퍼스에 없어서 못 맞히는" 인위적 손실이 생기지 않게 한다.
      3) distractor(train 단락)는 정답이 될 수 없는 오답 후보로만 들어가
         코퍼스를 키워 dense의 변별 난이도를 높인다.
    """
    # 안정적 id 부여 — 등장 순서 보존(dict 삽입순).
    context_to_id: dict[str, str] = {}

    def _id_of(ctx: str) -> str:
        if ctx not in context_to_id:
            context_to_id[ctx] = f"c{len(context_to_id):05d}"
        return context_to_id[ctx]

    # 1) 질의 확정 — validation 질문 앞쪽 n_queries개(중복 제거). gold 단락 동시 수집.
    queries: list[dict] = []
    seen_q: set[str] = set()
    gold_contexts: list[str] = []  # 질의들의 정답 단락(코퍼스 필수 포함)
    seen_gold: set[str] = set()
    for rec in records:
        if len(queries) >= n_queries:
            break
        q = rec["question"]
        if q in seen_q:
            continue
        seen_q.add(q)
        ctx = rec["context"]
        gold_id = _id_of(ctx)
        if ctx not in seen_gold:
            seen_gold.add(ctx)
            gold_contexts.append(ctx)
        queries.append({"question": q, "gold_id": gold_id})

    # 2) 코퍼스 구성 — gold 우선, 그다음 나머지 validation 단락, 마지막 train distractor.
    corpus_contexts: list[str] = list(gold_contexts)  # gold 무조건 포함
    corpus_ctx_set: set[str] = set(gold_contexts)

    def _add_until_full(source):
        for ctx in source:
            if len(corpus_contexts) >= n_contexts:
                return
            if ctx in corpus_ctx_set:
                continue
            corpus_ctx_set.add(ctx)
            corpus_contexts.append(ctx)

    # 나머지 validation 고유 단락(질의에 안 쓰인 것 포함) → 자연스러운 동일 도메인 후보.
    _add_until_full(rec["context"] for rec in records)
    # 코퍼스 목표 미달이면 train distractor로 채워 난이도↑.
    _add_until_full(distractors)

    corpus_ids = [_id_of(ctx) for ctx in corpus_contexts]
    corpus_texts = corpus_contexts
    return corpus_ids, corpus_texts, queries


# ---------------------------------------------------------------------------
# 2) 인덱싱 — bge-m3 → 별도 Chroma + BM25(공백/kiwi) 두 버전
# ---------------------------------------------------------------------------
def build_dense_index(corpus_ids: list[str], corpus_texts: list[str], round_idx: int = 0):
    """
    코퍼스 단락을 bge-m3로 임베딩해 별도 Chroma 경로에 인덱싱한다.
    볼트 인덱스(repo chroma/)와 완전 분리된 /tmp 경로를 매 실행 시 재구축.

    라운드별 고유 하위경로(round{n})를 사용한다. 포화 재측정으로 같은 프로세스에서
    인덱스를 재구축할 때, 이전 라운드의 Chroma persistent 클라이언트가 동일 경로를
    잡고 있어 "readonly database" 충돌이 나는 것을 피하기 위함이다.
    """
    from langchain_chroma import Chroma

    round_path = KORQUAD_CHROMA_PATH / f"round{round_idx}"
    # 이전 실행 잔여물 제거 → 깨끗한 인덱스 보장(누적 방지).
    if round_path.exists():
        shutil.rmtree(round_path)
    round_path.mkdir(parents=True, exist_ok=True)

    # 임베딩은 기존 팩토리 재사용하되, 경로/컬렉션만 벤치마크 전용으로 격리.
    cfg = Config()  # bge-m3, 로컬 Ollama 설정 상속
    embeddings = create_embeddings(cfg)

    vs = Chroma(
        collection_name=KORQUAD_COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(round_path),
        collection_metadata={"hnsw:space": "cosine"},  # 정규화 임베딩에 cosine
    )

    # 배치 임베딩 — 단락 본문을 문서로, id는 corpus_ids로 매핑.
    print(f"[인덱싱] bge-m3 임베딩 {len(corpus_texts)}개 단락 → {round_path} ...")
    BATCH = 200
    for start in range(0, len(corpus_texts), BATCH):
        end = min(start + BATCH, len(corpus_texts))
        vs.add_texts(
            texts=corpus_texts[start:end],
            ids=corpus_ids[start:end],
            metadatas=[{"doc_id": cid} for cid in corpus_ids[start:end]],
        )
        print(f"  ...{end}/{len(corpus_texts)} 임베딩 완료")
    return vs


def build_bm25(corpus_texts: list[str], tokenizer) -> BM25Okapi:
    """주어진 토크나이저로 BM25Okapi 인덱스를 구축한다."""
    tokenized = [tokenizer(t) for t in corpus_texts]
    return BM25Okapi(tokenized)


def whitespace_tokenize(text: str) -> list[str]:
    """공백 토크나이저 — off-the-shelf 하이브리드 기준선(소문자+공백 분리)."""
    return text.lower().split()


# ---------------------------------------------------------------------------
# 3) 검색 — dense / hybrid(공백) / hybrid(kiwi)
# ---------------------------------------------------------------------------
def dense_ranking(vs, query: str, k: int) -> list[str]:
    """dense 단독: Chroma 유사도 상위 k개 doc_id 리스트(순위순)."""
    results = vs.similarity_search_with_score(query, k=k)
    return [doc.metadata["doc_id"] for doc, _ in results]


def bm25_ranking(bm25: BM25Okapi, corpus_ids: list[str], tokenizer, query: str, k: int) -> list[str]:
    """BM25 단독: 점수 상위 k개 doc_id 리스트(순위순)."""
    scores = bm25.get_scores(tokenizer(query))
    # 점수 내림차순 정렬 인덱스 상위 k.
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [corpus_ids[i] for i in ranked_idx]


def rrf_fuse(dense_ids: list[str], bm25_ids: list[str], k: int) -> list[str]:
    """
    dense·BM25 순위 리스트를 RRF로 융합해 상위 k개 doc_id 반환.
    retrieval.py와 동일한 가중 RRF: score = w_dense/(K+rank) + w_bm25/(K+rank).
    """
    dense_rank = {doc_id: r for r, doc_id in enumerate(dense_ids, start=1)}
    bm25_rank = {doc_id: r for r, doc_id in enumerate(bm25_ids, start=1)}
    all_ids = set(dense_rank) | set(bm25_rank)
    fused: dict[str, float] = {}
    for doc_id in all_ids:
        score = 0.0
        if doc_id in dense_rank:
            score += DENSE_WEIGHT / (RRF_K + dense_rank[doc_id])
        if doc_id in bm25_rank:
            score += BM25_WEIGHT / (RRF_K + bm25_rank[doc_id])
        fused[doc_id] = score
    ranked = sorted(fused, key=lambda d: fused[d], reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# 4) 지표 — Hit@k, MRR@k
# ---------------------------------------------------------------------------
def evaluate(ranking_fn, queries: list[dict]) -> dict:
    """
    ranking_fn(query, k) → 순위순 doc_id 리스트.
    Hit@5, Hit@10, MRR@10 계산. 단락(문서) 단위 정답 매칭.
    """
    max_k = max(max(K_LIST), MRR_K)
    hits = {k: 0 for k in K_LIST}
    rr_sum = 0.0
    for q in queries:
        ranked = ranking_fn(q["question"], max_k)
        gold = q["gold_id"]
        # Hit@k
        for k in K_LIST:
            if gold in ranked[:k]:
                hits[k] += 1
        # MRR@k — 첫 정답의 역순위.
        for rank, doc_id in enumerate(ranked[:MRR_K], start=1):
            if doc_id == gold:
                rr_sum += 1.0 / rank
                break
    n = len(queries)
    return {
        "Hit@5": hits[5] / n,
        "Hit@10": hits[10] / n,
        "MRR@10": rr_sum / n,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run_once(
    n_contexts: int,
    n_queries: int,
    records: list[dict],
    distractors: list[str],
    round_idx: int = 0,
) -> tuple[dict, int, int]:
    """주어진 서브셋 규모로 3방식 평가를 1회 수행하고 결과 dict를 반환한다."""
    corpus_ids, corpus_texts, queries = build_subset(records, distractors, n_contexts, n_queries)
    print(
        f"\n[서브셋] 고유 단락(코퍼스)={len(corpus_texts)}개, "
        f"질의(질문)={len(queries)}개 (요청: 코퍼스 {n_contexts}, 질의 {n_queries})"
    )

    # dense 인덱스 (bge-m3, 별도 Chroma). 라운드별 고유 경로.
    vs = build_dense_index(corpus_ids, corpus_texts, round_idx=round_idx)

    # BM25 두 버전.
    print("[인덱싱] BM25(공백) 구축 ...")
    bm25_ws = build_bm25(corpus_texts, whitespace_tokenize)
    print("[인덱싱] BM25(kiwi) 구축 ...")
    bm25_kw = build_bm25(corpus_texts, kiwi_tokenize)

    # 3방식 ranking 함수.
    def f_dense(query, k):
        return dense_ranking(vs, query, k)

    def f_hybrid_ws(query, k):
        d = dense_ranking(vs, query, k)
        b = bm25_ranking(bm25_ws, corpus_ids, whitespace_tokenize, query, k)
        return rrf_fuse(d, b, k)

    def f_hybrid_kw(query, k):
        d = dense_ranking(vs, query, k)
        b = bm25_ranking(bm25_kw, corpus_ids, kiwi_tokenize, query, k)
        return rrf_fuse(d, b, k)

    print("[평가] dense 단독 ...")
    m_dense = evaluate(f_dense, queries)
    print("[평가] dense + 공백 BM25 (RRF) ...")
    m_ws = evaluate(f_hybrid_ws, queries)
    print("[평가] dense + kiwi BM25 (RRF) ...")
    m_kw = evaluate(f_hybrid_kw, queries)

    return (
        {"dense": m_dense, "hybrid_ws": m_ws, "hybrid_kw": m_kw},
        len(corpus_texts),
        len(queries),
    )


def print_table(results: dict, n_corpus: int, n_queries: int) -> None:
    """3방식 비교표 출력."""
    print("\n" + "=" * 72)
    print(f"KorQuAD v1 검색 벤치마크 — 코퍼스 {n_corpus}단락 / 질의 {n_queries}문항")
    print("출처: KorQuAD/squad_kor_v1 (validation) / 단락 단위 정답 매칭")
    print("=" * 72)
    header = f"{'방식':<28}{'Hit@5':>10}{'Hit@10':>10}{'MRR@10':>10}"
    print(header)
    print("-" * 72)
    labels = {
        "dense": "dense 단독",
        "hybrid_ws": "dense + 공백 BM25 (RRF)",
        "hybrid_kw": "dense + kiwi BM25 (RRF)",
    }
    for key in ("dense", "hybrid_ws", "hybrid_kw"):
        m = results[key]
        print(
            f"{labels[key]:<26}{m['Hit@5']*100:>9.2f}%"
            f"{m['Hit@10']*100:>9.2f}%{m['MRR@10']:>10.4f}"
        )
    print("=" * 72)


def main() -> None:
    print("KorQuAD v1 하이브리드 vs dense 벤치마크 시작")
    print(f"RRF 가중치: dense={DENSE_WEIGHT}, bm25={BM25_WEIGHT}, K={RRF_K}")
    print(f"임베딩: bge-m3 (로컬 Ollama) / Chroma 경로(격리): {KORQUAD_CHROMA_PATH}\n")

    records, distractors = _load_korquad_records()
    # 코퍼스 상한 = validation 고유 단락 + train distractor 풀 전체.
    max_corpus = len(set(r["context"] for r in records)) + len(distractors)

    # 1차 평가 — 요청 규모.
    n_ctx, n_q = N_CONTEXTS, N_QUERIES
    round_idx = 0
    results, n_corpus, n_queries = run_once(n_ctx, n_q, records, distractors, round_idx)
    print_table(results, n_corpus, n_queries)

    # --- 정직성: dense 포화(Hit@5 100%) 시 코퍼스↑로 난이도 상향 재시도 ---
    saturated = results["dense"]["Hit@5"] >= 1.0
    escalations = 0
    MAX_ESCALATIONS = 3
    while saturated and escalations < MAX_ESCALATIONS and n_ctx < max_corpus:
        escalations += 1
        n_ctx = min(n_ctx * 3, max_corpus)  # 3배씩 키워 빠르게 포화 탈출.
        print(
            f"\n[정직성] dense Hit@5=100% (포화). 코퍼스를 {n_ctx}단락으로 키워 "
            f"난이도를 높이고 재측정합니다 (시도 {escalations}/{MAX_ESCALATIONS}, 상한 {max_corpus})."
        )
        round_idx += 1
        results, n_corpus, n_queries = run_once(n_ctx, n_q, records, distractors, round_idx)
        print_table(results, n_corpus, n_queries)
        saturated = results["dense"]["Hit@5"] >= 1.0

    # --- 최종 정직한 결론 ---
    print("\n[결론]")
    d5 = results["dense"]["Hit@5"]
    print(f"- 서브셋 규모: 코퍼스 {n_corpus}단락, 질의 {n_queries}문항.")
    if d5 >= 1.0:
        print(
            f"- dense Hit@5={d5*100:.1f}% — 가능한 코퍼스 확대 후에도 포화가 깨지지 않았다. "
            "이 데이터셋/규모에서는 기법 우위 측정이 어렵다(사실 그대로 보고)."
        )
    else:
        print(f"- dense Hit@5={d5*100:.2f}% < 100% — 포화 깨짐 확인. 기법 비교 유효.")

    dense_mrr = results["dense"]["MRR@10"]
    ws_mrr = results["hybrid_ws"]["MRR@10"]
    kw_mrr = results["hybrid_kw"]["MRR@10"]
    print(f"- MRR@10: dense={dense_mrr:.4f}, 공백하이브리드={ws_mrr:.4f}, kiwi하이브리드={kw_mrr:.4f}")
    # kiwi vs dense
    if kw_mrr > dense_mrr:
        print(f"  · kiwi 하이브리드가 dense 대비 MRR +{kw_mrr - dense_mrr:.4f} (이득 있음).")
    elif kw_mrr < dense_mrr:
        print(f"  · kiwi 하이브리드가 dense 대비 MRR {kw_mrr - dense_mrr:.4f} (오히려 손해 — 그대로 보고).")
    else:
        print("  · kiwi 하이브리드와 dense MRR 동률.")
    # kiwi vs 공백
    if kw_mrr > ws_mrr:
        print(f"  · kiwi 하이브리드가 공백 하이브리드 대비 MRR +{kw_mrr - ws_mrr:.4f} (형태소 분석 이득).")
    elif kw_mrr < ws_mrr:
        print(f"  · kiwi 하이브리드가 공백 하이브리드 대비 MRR {kw_mrr - ws_mrr:.4f} (이득 없음 — 그대로 보고).")
    else:
        print("  · kiwi 하이브리드와 공백 하이브리드 MRR 동률.")

    print(f"\n[격리 확인] 벤치마크 Chroma 경로={KORQUAD_CHROMA_PATH} (볼트 chroma/ 미접촉).")


if __name__ == "__main__":
    sys.exit(main())
