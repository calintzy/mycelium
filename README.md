# mycelium

> 균사체(mycelium)는 흩어진 균사들이 땅속에서 연결되어 양분을 주고받는 거대한 지하 네트워크다.
> 이 프로젝트는 흩어진 마크다운 노트를 검색·그래프·RAG로 연결해, 지식이 서로를 보강하며
> 자라는 지하 네트워크처럼 동작하게 한다.

**어떤 마크다운 폴더에도 동작하는** 로컬 우선 하이브리드 검색 + RAG Q&A 도구.
질문하면 LLM이 노트를 근거로 답하고, **출처 노트까지 함께** 보여준다.
Obsidian에 의존하지 않는다 — `.md` 파일이 든 디렉토리 하나면 충분하다.

리포에는 완전 가공한 합성 코퍼스 `sample_vault/`(15개 공개 노트)가 동봉되어,
클론 즉시 인덱싱·검색·평가를 재현할 수 있다. 자기 노트를 쓰려면 `VAULT_PATH`만 지정한다.

---

## 특징
- **하이브리드 검색** — 의미검색(dense, BGE-M3) + 키워드(BM25)를 RRF로 융합.
- **한국어 제대로** — BM25에 kiwipiepy 형태소 토크나이저 주입(공백 분리의 한국어 함정 회피).
- **거리 게이트** — 최상위 dense 유사도가 임계 미만이면 LLM 호출 전에 "근거 없음" 확정(할루시네이션 방어).
- **완전 로컬** — 임베딩·생성 모두 Ollama. 데이터가 외부로 나가지 않음.
- **GraphRAG** — 위키링크·LLM 엔티티로 그래프를 짓고, 그래프 근접·커뮤니티 요약을 RRF 신호로 통합.
- **distill 컴파운딩** — 좋은 Q&A를 위키노트로 정제·누적해 쓸수록 코퍼스가 좋아진다.
- **검색 품질 평가** — grep vs 의미 vs 하이브리드를 같은 골드셋으로 Hit@k/MRR 비교.
- **MCP 서버** — Claude Code가 직접 볼트를 의미검색.

---

## 아키텍처 요약

의존 방향: `interfaces → pipeline → adapters → core` (클린 아키텍처 지향).

```
interfaces/   CLI(typer) + MCP 서버(FastMCP) — 입출력만
pipeline/     ingestion(로드·청킹·임베딩·적재) / retrieval(하이브리드) /
              generation(RAG) / graph(백본·커뮤니티·요약) / distill
adapters/     embedding / llm / vectorstore / graph_store — 로컬↔클라우드 교체 지점
core/         도메인 모델·설정 — 외부 의존 최소
```

데이터 흐름: `.md → 헤더 단위 청킹(+과대 섹션 2차 분할) → BGE-M3 임베딩 → Chroma 적재`
+ BM25 인덱스. 질의 시 dense·BM25(+그래프·요약) 순위를 RRF로 융합 → 거리 게이트 →
로컬 LLM이 출처 인용 답변 생성.

설계의 단일 진실: [`docs/DESIGN.md`](docs/DESIGN.md) (D-1~D-10),
[`docs/DESIGN_GRAPHRAG.md`](docs/DESIGN_GRAPHRAG.md) (Phase 7, D-11~D-17).

---

## 결과

### 자체 볼트 골드셋 (방향성 지표 — 표본 작음)

| 방식 | Hit@5 |
|---|---|
| grep (키워드만) | 77.8% |
| 하이브리드 (dense + kiwi BM25 + RRF) | 100% |

grep은 무순위라 MRR 미정의(DESIGN §7). 작은 골드셋은 통계적 벤치마크가 아니라
방향성 지표다 — 수치를 과장하지 않는다.

### KorQuAD v1 독립 벤치마크 (포화되지 않은 공개 한국어 셋)

자체 볼트는 dense가 이미 Hit@5 100%라 기법 우위가 드러나지 않는다.
포화를 깨기 위해 공개 한국어 QA 셋(KorQuAD v1)에서 같은 코퍼스로 3방식을 비교했다.

| 방식 | Hit@5 |
|---|---|
| dense 단독 | 97.0% |
| dense + 공백 BM25 (off-the-shelf 하이브리드) | 96.8% |
| dense + kiwi BM25 (우리 하이브리드) | **98.0%** |

핵심: **공백 BM25는 dense보다 오히려 낮다(96.8%)** — 한국어를 공백으로만 자르면
키워드 신호가 노이즈가 된다. **kiwi 형태소 토크나이저를 넣어야(98.0%)** 하이브리드가
dense를 실제로 이긴다(D-8). `benchmarks/korquad_bench.py`로 재현 가능.

> 정직성: 결과는 유불리와 무관하게 그대로 보고한다. 그래프 신호는 이 코퍼스에선
> 검색 Hit@k를 키우지 못해 RRF 가중 0으로 두되, 멀티홉·조망 "능력"으로만 쓴다
> (DESIGN_GRAPHRAG §8).

---

## 빠른 시작

### 1. 설치
```bash
git clone <repo-url> mycelium
cd mycelium
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

### 2. Ollama + 모델 준비
```bash
# https://ollama.com 설치 후
ollama serve
ollama pull bge-m3        # 임베딩 (1.2GB)
ollama pull qwen2.5:14b   # 생성 LLM (9GB)
```

### 3. 인덱싱 (기본 볼트 = 동봉 sample_vault/)
```bash
.venv/bin/python -m mycelium index
```

### 4. 검색 · 질문 · 평가 · 그래프
```bash
.venv/bin/python -m mycelium search "하이브리드 검색에서 RRF" --k 3
.venv/bin/python -m mycelium ask "스테이크 레스팅을 왜 하나?"
.venv/bin/python -m mycelium eval          # grep vs 의미 vs 하이브리드 비교표
.venv/bin/python -m mycelium graph-build   # 그래프 + 커뮤니티 + 요약 (LLM 호출)
```

`myco` 명령도 `python -m mycelium`과 동일하게 동작한다(editable 설치 시).

### 자기 노트로 바꾸기
```bash
VAULT_PATH=~/MyNotes .venv/bin/python -m mycelium index
```
실볼트 인덱스(`chroma/`)·그래프(`graph/`)는 `.gitignore` 대상이라 커밋되지 않는다(D-7).

---

## 인터페이스
- **CLI(주)** — `index` / `search` / `ask` / `eval` / `graph-build` / `distill` / `agentic` / `serve`.
- **MCP(선택)** — Claude Code 연동. 등록법은 [`docs/MCP_SETUP.md`](docs/MCP_SETUP.md).

---

## 스택
Python 3.12 · LangChain 1.3.x · Ollama(`bge-m3` + `qwen2.5:14b`) · Chroma ·
rank_bm25 + kiwipiepy · networkx + igraph + leidenalg · MCP

---

## 프라이버시 (D-7, deny-by-default)
공개 리포에는 합성 코퍼스 `sample_vault/`만 포함된다. 자기 볼트를 인덱싱한
결과물(`chroma/`·`graph/`)과 개인 골드셋(`*.local.yaml`)은 커밋되지 않는다.
`index --public-only`는 frontmatter `public: true` 노트만 별도 경로로 인덱싱한다.

---

## 라이선스
MIT — [`LICENSE`](LICENSE) 참고.
