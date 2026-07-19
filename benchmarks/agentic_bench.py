"""
agentic_bench.py — Claude Code 에이전틱 검색 벤치마크

목적
----
"Claude Code한테 볼트 검색을 그냥 시키면?" — 정확도·지연·비용 실측.

방식
----
goldset.local.yaml의 각 질의에 대해 headless claude CLI를 호출한다:
  claude -p "<프롬프트>" --allowedTools "Grep,Glob,Read" --output-format json

프롬프트: 볼트 경로를 명시하고, 해당 폴더의 마크다운 노트에서
          가장 관련 있는 상위 5개 파일명을 JSON 배열로만 반환하도록 지시.
          (읽기 전용 도구만 허용 — 볼트 파일 수정 절대 금지)

채점
----
반환 파일명 5개를 goldset 정답과 매칭한다.
evaluate.py의 _first_match_rank 로직과 동일 기준 (파일명 정규화 포함).
질의별 벽시계 시간 + usage 토큰을 기록한다.

실패 처리
---------
JSON 파싱 실패·타임아웃(질의당 최대 180초)은 miss로 집계하되 실패 건수를 별도 보고한다.
조용히 버리지 않는다.

표본 제한
---------
전 질의 실행하되 goldset이 40질의를 넘으면 앞 40개만 실행하고 표본 수를 명시한다.

격리 원칙
---------
- 볼트 파일 수정 금지: --allowedTools "Grep,Glob,Read" 로 읽기 전용 강제
- 실볼트는 claude가 읽기만 하며 mycelium 인덱스(chroma/)에는 접근하지 않음
- 기존 src/ 코드는 읽기만 한다
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

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

MAX_SAMPLES = 40          # goldset > 40 질의면 앞 40개만
QUERY_TIMEOUT_SEC = 180   # 질의당 최대 대기 시간

K = 5
MRR_K = 10


# ---------------------------------------------------------------------------
# goldset 로딩
# ---------------------------------------------------------------------------
def load_goldset(path: Path) -> list[dict]:
    """goldset.local.yaml을 로드해 질의 리스트를 반환한다."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("questions", [])


# ---------------------------------------------------------------------------
# 파일명 정규화 — evaluate.py의 기준과 동일
# ---------------------------------------------------------------------------
def _normalize(src: str) -> str:
    """볼트 상대경로 정규화: 앞뒤 공백·슬래시 제거, 백슬래시 통일."""
    return src.strip().lstrip("/").replace("\\", "/")


# ---------------------------------------------------------------------------
# claude headless 호출
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """\
아래 폴더의 마크다운(.md) 노트에서 다음 질의와 가장 관련 있는 노트 파일명 상위 5개를 \
관련도 순 JSON 배열로만 반환하라. 파일명은 볼트 루트 기준 상대경로(예: "04-AI-TOOLS/wiki/foo.md")로, \
설명 없이 JSON 배열만 출력할 것.

볼트 폴더: {vault_path}

질의: {query}"""


def call_claude(query: str, vault_path: Path) -> tuple[list[str], int, int, str | None]:
    """
    headless claude로 볼트 검색을 수행한다.

    반환
    ----
    (ranked_files, input_tokens, output_tokens, error_msg)
    - ranked_files: 반환된 상위 5개 파일 상대경로 (정규화됨). 실패 시 빈 리스트.
    - input_tokens, output_tokens: usage 토큰 (미확인 시 0).
    - error_msg: 성공 시 None, 실패 시 오류 메시지.
    """
    prompt = _PROMPT_TEMPLATE.format(vault_path=str(vault_path), query=query)

    cmd = [
        "claude",
        "-p", prompt,
        "--allowedTools", "Grep,Glob,Read",
        "--output-format", "json",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=QUERY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        return [], 0, 0, f"타임아웃 ({QUERY_TIMEOUT_SEC}초 초과)"
    except FileNotFoundError:
        return [], 0, 0, "claude CLI를 찾을 수 없음 (PATH 확인 필요)"

    if result.returncode != 0:
        stderr_snippet = result.stderr.strip()[:200] if result.stderr else ""
        return [], 0, 0, f"claude 종료코드={result.returncode} stderr={stderr_snippet!r}"

    # --output-format json 출력 파싱
    raw = result.stdout.strip()
    if not raw:
        return [], 0, 0, "빈 출력"

    # claude --output-format json은 여러 JSON 오브젝트를 줄 단위로 출력하거나
    # 단일 오브젝트를 출력할 수 있음. 마지막 result 오브젝트를 찾는다.
    input_tokens = 0
    output_tokens = 0
    response_text = ""

    # JSON Lines 형식 시도 (줄별 파싱)
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # result 타입의 오브젝트에서 텍스트 추출
        if isinstance(obj, dict):
            if obj.get("type") == "result":
                response_text = obj.get("result", "") or ""
                usage = obj.get("usage", {})
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)
            elif "result" in obj and "type" not in obj:
                # 단일 오브젝트 형식 fallback
                response_text = str(obj.get("result", ""))
                usage = obj.get("usage", {})
                if isinstance(usage, dict):
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

    # response_text가 없으면 raw 전체를 응답으로 간주
    if not response_text:
        response_text = raw

    # 응답에서 JSON 배열 추출 (markdown 코드블록 포함 대응)
    # 패턴: [ "...", "..." ] 형태의 JSON 배열
    array_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
    if not array_match:
        return [], input_tokens, output_tokens, f"JSON 배열 없음 in 응답: {response_text[:200]!r}"

    try:
        file_list = json.loads(array_match.group())
    except json.JSONDecodeError as e:
        return [], input_tokens, output_tokens, f"JSON 파싱 실패: {e} / 원문: {array_match.group()[:200]!r}"

    if not isinstance(file_list, list):
        return [], input_tokens, output_tokens, f"배열이 아닌 타입: {type(file_list)}"

    # 정규화 + 문자열만 추출
    ranked = [_normalize(str(f)) for f in file_list if isinstance(f, str)]
    return ranked, input_tokens, output_tokens, None


# ---------------------------------------------------------------------------
# 평가 실행
# ---------------------------------------------------------------------------
def run_agentic_bench(questions: list[dict]) -> dict:
    """
    goldset 질의 전체(최대 MAX_SAMPLES)에 대해 claude 에이전틱 검색을 실행하고
    Hit@5/MRR + 지연/토큰/실패 통계를 반환한다.
    """
    n_total = len(questions)
    if n_total > MAX_SAMPLES:
        print(f"[샘플링] goldset {n_total}개 > {MAX_SAMPLES}개 제한 → 앞 {MAX_SAMPLES}개만 평가")
        questions = questions[:MAX_SAMPLES]
    n = len(questions)

    hits5 = 0
    rr_sum = 0.0
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    failures: list[dict] = []

    print(f"\n[에이전틱 벤치] claude headless 검색 시작 ({n}개 질의, 최대 {QUERY_TIMEOUT_SEC}초/질의)")
    print(f"  볼트: {VAULT_PATH}")
    print(f"  허용 도구: Grep, Glob, Read (읽기 전용)")

    for i, q in enumerate(questions, start=1):
        query_text = q["question"]
        expected = [_normalize(s) for s in q["expected_sources"]]
        category = q.get("category", "")

        print(f"\n[{i:2d}/{n}] ({category}) {query_text[:60]}")

        t0 = time.time()
        ranked, in_tok, out_tok, err = call_claude(query_text, VAULT_PATH)
        elapsed = time.time() - t0

        total_latency += elapsed
        total_input_tokens += in_tok
        total_output_tokens += out_tok

        if err is not None:
            print(f"  실패: {err}")
            failures.append({"idx": i, "query": query_text, "error": err})
            continue

        print(f"  반환: {ranked}")
        print(f"  정답: {expected}")
        print(f"  지연: {elapsed:.1f}초  토큰: in={in_tok} out={out_tok}")

        # Hit@5
        hit = any(e in ranked[:K] for e in expected)
        if hit:
            hits5 += 1
            print(f"  → Hit@5 ✓")
        else:
            print(f"  → miss")

        # MRR@10
        for rank, src in enumerate(ranked[:MRR_K], start=1):
            if src in expected:
                rr_sum += 1.0 / rank
                break

    n_success = n - len(failures)
    metrics = {
        "Hit@5": hits5 / n if n > 0 else 0.0,
        "MRR": rr_sum / n if n > 0 else 0.0,
        "avg_latency_sec": total_latency / n_success if n_success > 0 else 0.0,
        "avg_input_tokens": total_input_tokens / n_success if n_success > 0 else 0,
        "avg_output_tokens": total_output_tokens / n_success if n_success > 0 else 0,
        "n_queries": n,
        "n_failures": len(failures),
        "failures": failures,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    return metrics


# ---------------------------------------------------------------------------
# 결과 출력
# ---------------------------------------------------------------------------
def print_results(metrics: dict) -> None:
    n = metrics["n_queries"]
    n_fail = metrics["n_failures"]
    n_ok = n - n_fail

    print("\n" + "=" * 80)
    print("에이전틱 검색 벤치마크 결과 (Claude Code headless)")
    print("=" * 80)

    print(f"\n| 지표 | 값 |")
    print(f"|------|-----|")
    print(f"| Hit@5 | {metrics['Hit@5']*100:.1f}% |")
    print(f"| MRR@10 | {metrics['MRR']:.4f} |")
    print(f"| 평균 지연 (초) | {metrics['avg_latency_sec']:.1f}s (성공 {n_ok}건 기준) |")
    print(f"| 평균 입력 토큰 | {metrics['avg_input_tokens']:.0f} |")
    print(f"| 평균 출력 토큰 | {metrics['avg_output_tokens']:.0f} |")
    print(f"| 총 입력 토큰 | {metrics['total_input_tokens']} |")
    print(f"| 총 출력 토큰 | {metrics['total_output_tokens']} |")
    print(f"| 표본 수 | {n}질의 |")
    print(f"| 실패 건수 | {n_fail}건 |")

    if metrics["failures"]:
        print(f"\n[실패 상세 — {n_fail}건]")
        for f in metrics["failures"]:
            print(f"  #{f['idx']}: {f['query'][:50]} → {f['error']}")

    print("\n[기준선 비교 (기존 측정치)]")
    print("| 방식 | 모델 | Hit@5 | MRR | 비고 |")
    print("|------|------|-------|-----|------|")
    print("| dense 단독 | bge-m3 | 100% | — | Mycelium 기존 측정 (포화) |")
    print("| kiwi 하이브리드 | bge-m3 | 100% | — | Mycelium 기존 측정 (포화) |")
    print("| 에이전틱 | Claude Code | (이번 실험) | (이번 실험) | headless claude |")

    print("\n[격리 확인]")
    print(f"  볼트: {VAULT_PATH} (읽기 전용, 수정 없음)")
    print(f"  허용 도구: Grep, Glob, Read")
    print(f"  타임아웃: {QUERY_TIMEOUT_SEC}초/질의")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    print("에이전틱 검색 벤치마크 (agentic_bench.py) 시작")
    print(f"VAULT_PATH: {VAULT_PATH}")
    print(f"GOLDSET: {GOLDSET_PATH}")
    print(f"표본 상한: {MAX_SAMPLES}개 / 타임아웃: {QUERY_TIMEOUT_SEC}초/질의")

    if not GOLDSET_PATH.exists():
        print(f"[오류] goldset.local.yaml 없음: {GOLDSET_PATH}")
        sys.exit(1)

    if not VAULT_PATH.exists():
        print(f"[오류] VAULT_PATH 없음: {VAULT_PATH}")
        sys.exit(1)

    # claude CLI 확인
    check = subprocess.run(["which", "claude"], capture_output=True, text=True)
    if check.returncode != 0:
        print("[오류] claude CLI를 찾을 수 없음. PATH에 claude가 있는지 확인하세요.")
        sys.exit(1)
    print(f"[claude] {check.stdout.strip()}")

    questions = load_goldset(GOLDSET_PATH)
    print(f"[골드셋] {len(questions)}개 질의 로드됨")

    metrics = run_agentic_bench(questions)
    print_results(metrics)


if __name__ == "__main__":
    sys.exit(main())
