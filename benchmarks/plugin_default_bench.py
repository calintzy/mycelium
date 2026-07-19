"""
plugin_default_bench.py — 소형 임베딩 모델(all-minilm) 벤치마크

목적
----
옵시디언 플러그인들이 기본 탑재하는 소형 임베딩 모델 수준의 dense 검색 품질을 측정한다.
프록시 모델: Ollama `all-minilm` (all-MiniLM-L6-v2, 384차원, 소형·영어 중심).

⚠️ 프록시 가정 (결과 해석 시 필독)
  - all-minilm은 "플러그인 기본값"의 프록시다. 실제 플러그인마다 다른 모델을 씀:
      · Smart Connections: all-MiniLM-L6-v2 (이 모델과 동일)
      · Text Generator: 기본 OpenAI text-embedding-ada-002 (대형·클라우드 의존)
      · Local GPT: nomic-embed-text 또는 all-minilm
      · Copilot: 원격 API
  - 따라서 이 실험은 "가장 가벼운 플러그인 기본값 상당"을 측정하며, 대형 플러그인 모델
    대비 실제 품질은 더 좋을 수도 있다.
  - 한국어 코퍼스(KorQuAD) 성능은 영어 중심 모델이라 bge-m3(다국어) 대비 불리하다.

격리 원칙
  - 볼트 인덱스(repo chroma/)·실볼트 원본은 절대 수정하지 않음.
  - 모든 Chroma 인덱스는 /tmp 하위 별도 경로에 새로 구축하며, 실행 후 삭제하지 않아도
    볼트와 완전 분리된다.
  - 기존 src/ 코드는 읽기만 한다.

측정 2종
  1) KorQuAD: korquad_bench.py의 프로토콜을 재사용해 all-minilm dense 단독 Hit@5/Hit@10/MRR@10.
  2) 자체 볼트: goldset.local.yaml 질의로 실볼트를 /tmp Chroma에 인덱싱 후 Hit@5/MRR.

출력
  마크다운 표 (방식/모델/코퍼스/Hit@5/MRR/표본 수).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
# 기본값은 리포 동봉 sample_vault — 실볼트 측정 시 VAULT_PATH 환경변수로 지정한다.
VAULT_PATH = Path(os.environ.get("VAULT_PATH", str(Path(__file__).parent.parent / "sample_vault")))
_EVAL_DIR = Path(__file__).parent.parent / "src" / "mycelium" / "eval"
# 실골드셋(goldset.local.yaml, 미추적)이 없으면 샘플 골드셋으로 폴백 — 클론 즉시 실행 가능.
GOLDSET_PATH = _EVAL_DIR / "goldset.local.yaml"
if not GOLDSET_PATH.exists():
    GOLDSET_PATH = _EVAL_DIR / "goldset.yaml"

# KorQuAD 표본 수 — 기존 korquad_bench.py와 동일한 환경변수로 제어.
N_QUERIES = int(os.environ.get("KORQUAD_N_QUERIES", "500"))
N_CONTEXTS = int(os.environ.get("KORQUAD_N_CONTEXTS", "2000"))

# /tmp 격리 경로
PLUGIN_CHROMA_KORQUAD = Path("/tmp/plugin_bench_korquad_chroma")
PLUGIN_CHROMA_VAULT = Path("/tmp/plugin_bench_vault_chroma")

OLLAMA_MODEL = "all-minilm"
K_LIST = (5, 10)
MRR_K = 10

# 볼트 인덱싱 제외 디렉토리 — mycelium 기존 기준과 동일하게 유지
_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {".obsidian", ".omc", ".git", ".venv", "__pycache__", "00-INBOX"}
)


# ---------------------------------------------------------------------------
# Ollama all-minilm 확인 및 pull
# ---------------------------------------------------------------------------
def ensure_model() -> None:
    """all-minilm이 없으면 ollama pull로 설치한다."""
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    if OLLAMA_MODEL not in result.stdout:
        print(f"[모델] {OLLAMA_MODEL} 미설치 → ollama pull {OLLAMA_MODEL} 실행 중...")
        subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)
        print(f"[모델] {OLLAMA_MODEL} 설치 완료.")
    else:
        print(f"[모델] {OLLAMA_MODEL} 확인됨.")


# ---------------------------------------------------------------------------
# 임베딩 팩토리 (all-minilm 전용)
# ---------------------------------------------------------------------------
def _make_embeddings():
    """all-minilm OllamaEmbeddings 인스턴스를 반환한다."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=OLLAMA_MODEL, base_url="http://127.0.0.1:11434")


# ---------------------------------------------------------------------------
# 1) KorQuAD 벤치 — korquad_bench.py의 코퍼스/평가 프로토콜 재사용
# ---------------------------------------------------------------------------
def run_korquad_bench() -> dict:
    """
    KorQuAD 서브셋에서 all-minilm dense 단독 Hit@5/Hit@10/MRR@10을 측정한다.
    korquad_bench.py의 _load_korquad_records, build_subset, evaluate 로직을 재사용한다.
    """
    print("\n[KorQuAD 벤치] all-minilm dense 단독 측정 시작")

    # korquad_bench의 데이터셋 로더/서브셋 빌더/평가 함수 재사용
    sys.path.insert(0, str(Path(__file__).parent))
    from korquad_bench import _load_korquad_records, build_subset, evaluate  # 읽기 전용 재사용

    from langchain_chroma import Chroma

    records, distractors = _load_korquad_records()
    corpus_ids, corpus_texts, queries = build_subset(records, distractors, N_CONTEXTS, N_QUERIES)
    print(
        f"[서브셋] 코퍼스={len(corpus_texts)}단락, 질의={len(queries)}개 "
        f"(요청: {N_CONTEXTS}/{N_QUERIES})"
    )

    # 기존 경로 정리 후 새로 구축
    if PLUGIN_CHROMA_KORQUAD.exists():
        shutil.rmtree(PLUGIN_CHROMA_KORQUAD)
    PLUGIN_CHROMA_KORQUAD.mkdir(parents=True, exist_ok=True)

    embeddings = _make_embeddings()
    vs = Chroma(
        collection_name="plugin_korquad",
        embedding_function=embeddings,
        persist_directory=str(PLUGIN_CHROMA_KORQUAD),
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"[인덱싱] all-minilm 임베딩 {len(corpus_texts)}개 단락 → {PLUGIN_CHROMA_KORQUAD} ...")
    BATCH = 500  # all-minilm은 소형이라 배치를 크게 잡아도 됨
    for start in range(0, len(corpus_texts), BATCH):
        end = min(start + BATCH, len(corpus_texts))
        vs.add_texts(
            texts=corpus_texts[start:end],
            ids=corpus_ids[start:end],
            metadatas=[{"doc_id": cid} for cid in corpus_ids[start:end]],
        )
        print(f"  ...{end}/{len(corpus_texts)} 임베딩 완료")

    def f_dense(query: str, k: int) -> list[str]:
        results = vs.similarity_search_with_score(query, k=k)
        return [doc.metadata["doc_id"] for doc, _ in results]

    print("[평가] KorQuAD all-minilm dense 단독 ...")
    metrics = evaluate(f_dense, queries)
    print(f"  Hit@5={metrics['Hit@5']*100:.2f}%  Hit@10={metrics['Hit@10']*100:.2f}%  MRR@10={metrics['MRR@10']:.4f}")

    return {
        "metrics": metrics,
        "n_corpus": len(corpus_texts),
        "n_queries": len(queries),
    }


# ---------------------------------------------------------------------------
# 2) 자체 볼트 벤치 — goldset.local.yaml + 실볼트 /tmp 인덱싱
# ---------------------------------------------------------------------------
def _collect_vault_files(vault_path: Path) -> list[Path]:
    """볼트 .md 파일을 기존 mycelium 기준(제외 디렉토리 동일)으로 수집한다. 읽기 전용."""
    files: list[Path] = []
    for item in vault_path.rglob("*.md"):
        if any(part in _EXCLUDE_DIRS for part in item.parts):
            continue
        files.append(item)
    return files


def _normalize_source(src: str) -> str:
    """파일명 정규화 — 비교 시 앞뒤 공백·슬래시 정리."""
    return src.strip().lstrip("/")


def run_vault_bench() -> dict:
    """
    goldset.local.yaml의 질의로 실볼트를 all-minilm으로 /tmp Chroma에 인덱싱,
    dense 검색 Hit@5/MRR을 측정한다. 볼트 파일은 읽기만 하며 수정하지 않는다.

    평가 기준: evaluate.py의 _first_match_rank / _evaluate_ranking과 동일한 로직.
    파일명 매칭: 볼트 상대경로 기준 (goldset의 expected_sources와 동일 포맷).
    """
    print("\n[볼트 벤치] all-minilm dense + 실볼트 측정 시작")

    if not VAULT_PATH.exists():
        print(f"[오류] VAULT_PATH={VAULT_PATH} 존재하지 않음. 볼트 벤치를 건너뜁니다.")
        return {"error": "vault_not_found", "n_queries": 0}

    if not GOLDSET_PATH.exists():
        print(f"[오류] goldset.local.yaml={GOLDSET_PATH} 없음. 볼트 벤치를 건너뜁니다.")
        return {"error": "goldset_not_found", "n_queries": 0}

    import yaml
    from langchain_chroma import Chroma

    # goldset 로딩
    data = yaml.safe_load(GOLDSET_PATH.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    print(f"[골드셋] {len(questions)}개 질의 로드됨 (goldset.local.yaml)")

    # 볼트 파일 수집 (읽기 전용)
    vault_files = _collect_vault_files(VAULT_PATH)
    print(f"[볼트] {len(vault_files)}개 .md 파일 발견 (VAULT_PATH={VAULT_PATH})")

    if not vault_files:
        print("[오류] 볼트 파일이 없음. 볼트 벤치를 건너뜁니다.")
        return {"error": "no_vault_files", "n_queries": 0}

    # /tmp Chroma에 인덱싱 (기존 경로 정리 후 재구축)
    if PLUGIN_CHROMA_VAULT.exists():
        shutil.rmtree(PLUGIN_CHROMA_VAULT)
    PLUGIN_CHROMA_VAULT.mkdir(parents=True, exist_ok=True)

    embeddings = _make_embeddings()
    vs = Chroma(
        collection_name="plugin_vault",
        embedding_function=embeddings,
        persist_directory=str(PLUGIN_CHROMA_VAULT),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 볼트 파일 텍스트 + 메타데이터 (source = 볼트 상대경로)
    texts: list[str] = []
    sources: list[str] = []
    ids: list[str] = []
    for i, f in enumerate(vault_files):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(f.relative_to(VAULT_PATH))
        texts.append(content)
        sources.append(rel)
        ids.append(f"v{i:05d}")

    print(f"[인덱싱] all-minilm 임베딩 {len(texts)}개 파일 → {PLUGIN_CHROMA_VAULT} ...")
    BATCH = 200
    for start in range(0, len(texts), BATCH):
        end = min(start + BATCH, len(texts))
        vs.add_texts(
            texts=texts[start:end],
            ids=ids[start:end],
            metadatas=[{"source": sources[i]} for i in range(start, end)],
        )
        print(f"  ...{end}/{len(texts)} 임베딩 완료")

    # goldset 평가 — evaluate.py의 _first_match_rank 로직과 동일
    K = 5
    MRR_K_LOCAL = 10
    hits5 = 0
    rr_sum = 0.0
    n = len(questions)

    for q in questions:
        query_text = q["question"]
        expected = [_normalize_source(s) for s in q["expected_sources"]]

        # dense 검색 top-20 (노트 단위 dedup 후 top-K 평가)
        results = vs.similarity_search_with_score(query_text, k=20)
        ranked_sources: list[str] = []
        seen: set[str] = set()
        for doc, _ in results:
            src = _normalize_source(doc.metadata.get("source", ""))
            if src and src not in seen:
                seen.add(src)
                ranked_sources.append(src)

        # Hit@5
        hit = any(e in ranked_sources[:K] for e in expected)
        if hit:
            hits5 += 1

        # MRR@10
        for rank, src in enumerate(ranked_sources[:MRR_K_LOCAL], start=1):
            if src in expected:
                rr_sum += 1.0 / rank
                break

    metrics = {
        "Hit@5": hits5 / n,
        "MRR": rr_sum / n,
    }
    print(f"  Hit@5={metrics['Hit@5']*100:.1f}%  MRR={metrics['MRR']:.4f}")

    return {
        "metrics": metrics,
        "n_queries": n,
        "n_vault_files": len(vault_files),
    }


# ---------------------------------------------------------------------------
# 결과 출력 — 마크다운 표
# ---------------------------------------------------------------------------
def print_results(korquad_res: dict, vault_res: dict) -> None:
    print("\n" + "=" * 80)
    print("소형 임베딩 벤치마크 (all-minilm / Smart Connections 기본값 프록시)")
    print("⚠️  프록시 가정: all-minilm은 일부 플러그인 기본값과 동일하나 전부는 아님")
    print("=" * 80)

    print("\n| 방식 | 모델 | 코퍼스 | Hit@5 | Hit@10 | MRR | 표본 수 |")
    print("|------|------|--------|-------|--------|-----|---------|")

    if "error" not in korquad_res:
        m = korquad_res["metrics"]
        n_c = korquad_res["n_corpus"]
        n_q = korquad_res["n_queries"]
        print(
            f"| dense 단독 | all-minilm | KorQuAD {n_c}단락 "
            f"| {m['Hit@5']*100:.2f}% | {m['Hit@10']*100:.2f}% "
            f"| {m['MRR@10']:.4f} | {n_q}질의 |"
        )
    else:
        print(f"| KorQuAD | - | - | 오류: {korquad_res['error']} | - | - | - |")

    if "error" not in vault_res:
        m = vault_res["metrics"]
        n_q = vault_res["n_queries"]
        n_v = vault_res["n_vault_files"]
        print(
            f"| dense 단독 | all-minilm | 실볼트 {n_v}파일 "
            f"| {m['Hit@5']*100:.1f}% | N/A "
            f"| {m['MRR']:.4f} | {n_q}질의 |"
        )
    else:
        print(f"| 볼트 | - | - | 오류: {vault_res['error']} | - | - | - |")

    print("\n[기준선 비교 (기존 측정치)]")
    print("| 방식 | 모델 | 코퍼스 | Hit@5 | MRR | 비고 |")
    print("|------|------|--------|-------|-----|------|")
    print("| dense 단독 | bge-m3 | KorQuAD 2000단락 | 97.0% | — | Mycelium 기존 측정 |")
    print("| kiwi 하이브리드 | bge-m3 | KorQuAD 2000단락 | 98.0% | — | Mycelium 기존 측정 |")
    print("| dense 단독 | bge-m3 | 실볼트 | 100% | — | Mycelium 기존 측정 (포화) |")

    print("\n[격리 확인]")
    print(f"  KorQuAD Chroma: {PLUGIN_CHROMA_KORQUAD} (볼트 chroma/ 미접촉)")
    print(f"  볼트 Chroma: {PLUGIN_CHROMA_VAULT} (볼트 파일 읽기 전용, 수정 없음)")
    print(f"  실볼트 VAULT_PATH={VAULT_PATH}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("소형 임베딩 벤치마크 (plugin_default_bench.py) 시작")
    print(f"프록시 모델: {OLLAMA_MODEL} (all-MiniLM-L6-v2, 소형·영어 중심)")
    print(f"KorQuAD: 코퍼스 {N_CONTEXTS}단락, 질의 {N_QUERIES}개")
    print(f"VAULT_PATH: {VAULT_PATH}")

    ensure_model()

    korquad_res = run_korquad_bench()
    vault_res = run_vault_bench()

    print_results(korquad_res, vault_res)


if __name__ == "__main__":
    sys.exit(main())
