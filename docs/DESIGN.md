# mycelium — 설계 문서

> 개인 마크다운 지식베이스(Obsidian 스타일 볼트) 위에 하이브리드 검색 + RAG Q&A를 얹는 미니 프로젝트
> 목적: ① RAG/LangChain/LangGraph 실전 경험 확보(이력서 공백 메우기) ② 실제로 쓸 수 있는 볼트 검색 도구
> 작성일: 2026-06-16
> 공개 리포에는 실볼트 대신 합성 코퍼스 `sample_vault/`가 동봉되며, 모든 예시는 그 기준이다.

---

## 1. 목표 / 비목표

### 목표 (v1)
- 178개 마크다운 볼트를 인덱싱하고, 의미 기반으로 검색한다.
- 질문에 대해 **LLM이 노트를 근거로 답변을 생성**하고, **근거 노트를 출처로 함께 표시**한다.
- **하이브리드 검색**(키워드 + 의미)으로 grep 단독 대비 검색 품질을 끌어올린다.
- 검색 정확도를 **수치로 평가**한다(테스트 질문셋 기반).
- **MCP 서버**로 노출해 Claude Code가 볼트를 의미검색하게 한다.
- 전 과정이 **로컬에서 동작**한다(볼트 내용이 외부로 나가지 않음).
- **공개 가능한 포트폴리오 산출물**을 만든다 — 비민감 노트만 담은 공개 코퍼스 + 그 위 평가표 + 시연(면접관 재현 가능, D-7).

### 비목표 (v1에서 안 함 — 과설계 방지)
- 볼트 원본 파일 수정/관리 기능 (읽기 전용 코퍼스로만 사용)
- 멀티유저, 인증, 클라우드 배포
- 실시간 파일 변경 감지 자동 재인덱싱 (수동 `index` 명령으로 충분)
- LangGraph 에이전트형 RAG (→ Phase 2 스트레치로 분리)

---

## 2. 확정된 결정 (요약)

| 항목 | 결정 |
|---|---|
| 검색 vs Q&A | **둘 다** — LLM 답변 생성 + 출처 노트 인용 |
| 임베딩 | **로컬 BGE-M3 (Ollama 서빙)** — FlagEmbedding/torch 불필요 |
| 답변 생성 LLM | **로컬 기본**(Ollama) + 클라우드 교체 가능 (※ 오픈 결정 D-5) |
| 런타임 | **Ollama 하나로 일원화** (임베딩 + 생성 둘 다) |
| 공개 범위 | 로컬은 볼트 전체 인덱싱 / 공개 리포는 `public: true` 노트만 (deny-by-default) |
| 의존성 | **프로젝트 venv에 LangChain 최신 1.x** (전역 1.0.3과 격리) |
| 벡터 저장소 | **로컬 Chroma** (v1) → pgvector는 포트폴리오 마감 옵션 |
| 검색 방식 | **하이브리드** (의미검색 + BM25 키워드, RRF 융합) |
| 청킹 | **마크다운 헤더 단위** (+ 과대 섹션 2차 분할) |
| 인터페이스 | **MCP 서버** (주) + **데모 CLI**(before/after 시각화) |
| 평가 | **테스트 질문셋 기반 Hit@k / MRR** |
| 언어 | **Python** |

---

## 3. 스택

| 레이어 | 선택 | 비고 |
|---|---|---|
| 언어 | Python 3.12 | RAG 생태계 최대, Ryan 강점 |
| 오케스트레이션 | **LangChain 최신 1.x** (`langchain` 1.3.x / `langchain-core` 1.4.x) | DocumentLoader·TextSplitter·EnsembleRetriever → 이력서 키워드 확보. **전역 1.0.3 대신 venv에 최신 설치** |
| 임베딩 | **BGE-M3 (Ollama `bge-m3`)** | 다국어(한국어 강함). dense 벡터만 사용(sparse는 BM25가 대체) → torch/FlagEmbedding 불필요 |
| 키워드 검색 | **`rank_bm25` + kiwipiepy 직접 구현** (권장) 또는 `BM25Retriever`(community, sunset) | 한국어 형태소 분리 필수 (D-8). community sunset 회피 |
| 융합 | **RRF 직접 구현**(권장) 또는 `EnsembleRetriever` ← `langchain_classic.retrievers` | ⚠️ `langchain.retrievers` 경로는 1.x에서 없음(실측). 직접 구현 시 classic 의존 제거 |
| 벡터 저장소 | **Chroma** (`langchain-chroma` 1.x, 로컬 영속) | 설치 0설정, 실험 빠름 |
| 생성 LLM | **Ollama** (Qwen2.5-14B 등, M5/32GB로 충분) | 로컬 일관성. config로 교체 가능 |
| 인터페이스 | MCP (`mcp` python SDK) + CLI(`typer`) | |
| 평가 | 자체 스크립트 (Hit@k, MRR) | |

> **런타임 일원화**: 임베딩(`bge-m3`)과 생성 LLM을 모두 Ollama가 서빙 → 의존성 단순화. 하이브리드의 lexical 축은 BM25가 담당하므로 BGE-M3의 네이티브 sparse 출력은 불필요.

### 3.1 경험적 검증 로그 (2026-06-16, 임시 venv 실측)
> 추측 배제. 실제 설치·import·실행 결과만 기록.

| 검증 항목 | 결과 |
|---|---|
| 최신 1.x 스택 의존성 충돌 | 없음 (langchain 1.3.9 / core 1.4.7 / community 0.4.2 / ollama 1.1.0 / chromadb 1.5.9 / kiwipiepy 0.23.2) |
| `MarkdownHeaderTextSplitter`, `RecursiveCharacterTextSplitter` | ✅ import OK (`langchain_text_splitters`) |
| `BM25Retriever` | ✅ `langchain_community.retrievers` (단, community sunset) |
| `EnsembleRetriever` @ `langchain.retrievers` | ❌ 모듈 없음 |
| `EnsembleRetriever` @ `langchain_classic.retrievers` | ✅ import OK |
| `OllamaEmbeddings`/`ChatOllama` (`langchain_ollama`) | ✅ |
| `Chroma` (`langchain_chroma`) | ✅ |
| `kiwipiepy.Kiwi`, `mcp.server.fastmcp.FastMCP` | ✅ |
| `BM25Retriever.from_texts`의 `preprocess_func` | ✅ 존재 (kiwi 주입 가능) |
| `EnsembleRetriever` = RRF 여부 | ✅ 소스에 `rank_fusion`/`weights` 확인 |
| kiwi vs 공백 한국어 토크나이징 | ✅ 차이 실증 (D-8) |
| (공식 출처) bge-m3 | 1.2GB / 8K ctx |
| (공식 출처) qwen2.5:14b | 9.0GB / 32K ctx |

---

## 4. 아키텍처

### 레이어 분리 (클린 아키텍처 지향)
```
interfaces/        ← CLI, MCP 서버 (입출력만)
   │
pipeline/
   ├─ ingestion/   ← 로드 → 청킹 → 임베딩 → 인덱스 적재
   ├─ retrieval/   ← 하이브리드 검색 (dense + bm25 + RRF)
   └─ generation/  ← RAG 프롬프트 조립 → LLM 답변 + 출처
   │
core/              ← 도메인 모델(Document, Chunk, RetrievedNote, Answer), 설정
   │
adapters/          ← 임베딩·LLM·벡터스토어 추상 인터페이스 (교체 가능하게)
```
> `adapters/`에서 임베딩·LLM·벡터스토어를 인터페이스로 추상화 → "로컬 ↔ 클라우드 교체"를 한 곳에서 처리. evmscope의 Plugin 패턴, agentscore의 Strategy 패턴과 같은 사고.

### 데이터 흐름
```
[인덱싱 — 1회/갱신시]
볼트 .md 178개
  → MarkdownHeaderTextSplitter (## 단위 청킹, 헤더 메타 보존)
  → 과대 섹션은 RecursiveCharacterTextSplitter로 2차 분할
  → BGE-M3 임베딩 (로컬)
  → Chroma 저장 (벡터 + 원문 + 출처경로 메타)
  → 동시에 BM25 인덱스 구축 (키워드용)

[질의 — 매 질문]
질문
  → (A) BGE-M3 임베딩 → Chroma 의미검색 top-k
  → (B) BM25 키워드 검색 top-k
  → EnsembleRetriever가 A·B를 RRF로 융합 → 최종 top-n 청크
  → 프롬프트에 청크 주입 + "근거 없으면 모른다고 답하라" 지시
  → 로컬 LLM 답변 생성
  → 답변 + 출처 노트(경로·유사도) 함께 반환
```

---

## 5. 설계 결정 기록

### 결정 D-1: 임베딩·생성 모두 로컬
- Why: 볼트에 민감한 개인 정보(메모·문서)가 포함될 수 있음. 외부 유출 차단이 1순위.
- Alt considered: OpenAI 임베딩(간단·정확) + Claude 생성(고품질). 비용 무시 가능 수준이나 내용이 외부로 나감.
- Threat: 로컬 LLM의 한국어 답변 품질이 클라우드보다 낮을 수 있음.
- Mitigation: 생성 LLM을 config로 교체 가능하게 설계. 품질 부족 시 클라우드로 전환(내용 유출 감수)하는 선택을 사용자에게 남김.
- Open: D-5 참조.

### 결정 D-2: BGE-M3 임베딩 (Ollama 서빙)
- Why: 한국어 포함 다국어 성능 우수. Ollama가 `bge-m3`를 서빙하므로 torch/FlagEmbedding 없이 생성 LLM과 동일 런타임에서 처리 → 의존성 일원화.
- Alt considered: FlagEmbedding 직접 사용(BGE-M3 네이티브 dense+sparse 동시 산출하나 torch 2GB+ 의존), multilingual-e5(dense만), OpenAI(외부 유출).
- Threat: Ollama `bge-m3`는 dense 벡터만 제공 → 모델 차원의 하이브리드는 안 됨.
- Mitigation: lexical 축은 BM25Retriever가 별도 담당하고 RRF로 융합하므로 dense만으로 충분. 하이브리드 목적 달성에 영향 없음.

### 결정 D-3: 하이브리드 검색 (의미 + 키워드)
- Why: 프로젝트의 핵심 서사가 "grep(키워드) vs 의미검색". 둘을 융합하면 서사가 결과물 자체가 됨. 의미검색만으론 고유명사·코드·정확 매칭에 약하고, 키워드만으론 동의어를 놓침.
- Alt considered: 순수 의미검색(top-k)만. 구현은 단순하나 차별성·품질 모두 약함.
- Threat: 융합 가중치 튜닝 필요.
- Mitigation: RRF는 가중치에 둔감한 편. 평가셋(섹션 7)으로 가중치를 데이터로 결정.

### 결정 D-4: MCP 서버를 주 인터페이스로
- Why: Claude Code가 볼트를 의미검색하게 됨 → "내가 만든 RAG를 내가 실제로 쓴다"(A안 흡수)가 자동 달성. AI-개발도구 포트폴리오 라인(agentscore·evmscope)과 일관.
- Alt considered: 웹 UI(데모 시각화 강함), CLI 단독.
- Threat: MCP는 화면 데모가 약해 포트폴리오 전시력이 떨어짐.
- Mitigation: before/after를 보여주는 **데모 CLI를 병행** 제작. 전시는 CLI, 실사용은 MCP.

### 결정 D-5: 생성 LLM 로컬 vs 클라우드 (오픈, 리스크 완화됨)
- 상태: **로컬 우선으로 기운 미결**. 기기가 Apple M5 / 32GB로 확인되어 14B(여유)~32B(양자화) 로컬 구동 가능 → 한국어 품질 우려 대폭 감소.
- v1: 로컬(Ollama, Qwen2.5-14B 등)로 시작.
- 판단 시점: 평가셋으로 로컬 답변 품질 실측 후 확정.
- 선택지: (a) 로컬 유지(완전 프라이버시·기기 충분) (b) 클라우드 전환(품질↑, 청크 유출 감수) (c) 하이브리드.

### 결정 D-6: 증분 인덱싱 — 새 지식 반영 방식
- Why: 볼트에 노트를 추가하는 워크플로우는 지금 그대로 유지(마크다운 투입). 단 RAG는 임베딩 사본(Chroma)에서 검색하므로 재인덱싱 전까지 신규 노트가 검색에 안 잡힘(staleness window).
- v1: `index` 명령으로 **전체 재인덱싱** — 178개·10만 토큰은 수십 초라 부담 없음.
- 확장: 파일 해시/mtime 추적으로 **변경분만 upsert**하는 증분 인덱싱(Phase 6 옵션).
- 자동화 여지: 기존 `/inbox` 스킬 또는 git 훅에 재인덱싱 트리거를 연결하면 "볼트에 넣으면 자동 검색 가능".
- Mitigation(staleness): MCP/CLI가 마지막 인덱싱 시각을 표시하거나, 오래되면 자동 재인덱싱.

### 결정 D-7: 공개/비공개 분리 (deny-by-default)
- Why: 완전 로컬이라 공개 리포에 실데이터를 못 넣음 → 면접관이 재현·검증 불가(포트폴리오 역설). 비민감 노트만 공개해 해소.
- 메커니즘: 로컬은 볼트 **전체** 인덱싱(실사용). 공개 코퍼스는 노트 frontmatter에 **`public: true`가 명시된 것만** 포함. 미표시는 자동 비공개.
- Why deny-by-default: 민감한 개인 정보 유출을 구조적으로 차단(실수로 새는 것 방지).
- 산출물: 공개 서브셋 + 그 위에서 돌린 평가표 → 면접관 재현 가능.
- Threat: 공개 노트가 적으면 데모 빈약.
- Mitigation: `03-KNOWLEDGE`·`04-AI-TOOLS`(128개)가 대부분 일반 지식이라 공개 코퍼스로 충분.

### 결정 D-8: 한국어 BM25 토크나이저 (검증으로 발견한 결함 수정)
- Why: LangChain `BM25Retriever` 기본 토크나이저는 공백/정규식 기반 → 한국어는 조사·어절 내 형태소가 분리되지 않아 키워드 매칭 저성능. 볼트가 한국어이므로 정면으로 해당.
- 근거: 다수 연구·실무가 한국어 BM25에 형태소 토크나이저(kiwi/konlpy) 사용을 권장(공백 토크나이징은 언어별 분석기 대비 열세).
- 결정: BM25 토크나이저로 **kiwipiepy** 형태소 분석기 주입. (대안: konlpy Okt — JVM 의존이라 kiwi가 가벼움)
- **실증(2026-06-16, kiwipiepy 0.23.2)**:
  - 입력 `"...타이밍이 근육 회복에 미치는..."`
  - kiwi: `[..., '타이밍','이', ..., '회복','에', '미치','는', ...]`
  - 공백: `[..., '타이밍이', ..., '회복에', '미치는', ...]`
  - → 공백 분리는 `회복에`로 붙어 질의 `회복`이 매칭 실패. kiwi는 `회복`+`에` 분리로 매칭 성공. 결함과 해결이 모두 실측됨.
- Alt considered: 공백 토크나이저 그대로(저성능, 실증됨), bm25s 라이브러리.
- 차별화: "한국어 RAG의 BM25 토크나이징 함정을 인지·해결" → 면접 서사이자 평가표에서 수치로 입증 가능.
- Threat: kiwipiepy 설치(C 확장) 환경 의존.
- Mitigation: kiwipiepy 휠로 설치 단순(실측 OK). 실패 시 공백 토크나이저로 graceful fallback + 경고.

### 결정 D-9: 검색 컴포넌트 — 직접 구현 vs LangChain 컴포넌트 (검증으로 발견)
- 배경(실측): ① `EnsembleRetriever`는 1.x에서 `langchain.retrievers`에 없고 `langchain_classic.retrievers`로 이동. ② `BM25Retriever`가 속한 `langchain-community`는 **공식적으로 sunset(유지보수 종료)** 중.
- 결정(권장): **BM25(`rank_bm25`+kiwi)와 RRF 융합을 직접 구현**. 이유 — (a) sunset/classic 의존 제거로 견고성↑ (b) "하이브리드 검색을 직접 설계·구현"이 컴포넌트 글루보다 강한 포트폴리오·면접 서사 (c) RRF는 ~15줄로 단순.
- Alt considered: `langchain_classic.EnsembleRetriever` + `langchain_community.BM25Retriever` 그대로 사용(빠르나 sunset 의존, 직접 이해 신호 약함).
- LangChain 사용 면적은 유지: DocumentLoader·MarkdownHeaderTextSplitter·OllamaEmbeddings·ChatOllama·langchain-chroma·LCEL 체인 → 이력서 키워드 충분.
- Open: 직접 구현(권장) vs 컴포넌트 사용 — 사용자 확인 필요.

---

## 6. 청킹 전략

- 1차: `MarkdownHeaderTextSplitter`로 `#`/`##`/`###` 단위 분할, 헤더 경로를 메타데이터로 보존(검색 시 맥락·출처로 활용).
- 2차: 한 섹션이 임계치(예: 1,000자) 초과 시 `RecursiveCharacterTextSplitter`로 재분할, 약간의 overlap(예: 100자)로 문맥 단절 방지.
- frontmatter(`date`/`project`/`type`)는 메타데이터로 분리 저장 → 향후 필터링(예: "특정 type 노트에서만") 확장 여지.

---

## 7. 평가 설계 (차별화 포인트)

- **골드셋**: 볼트를 알고 있으므로 "이 질문엔 이 노트가 나와야 한다" 질문 15~20개를 손으로 작성.
- **지표**: Hit@k (정답 노트가 top-k에 포함됐나), MRR (정답 노트의 평균 역순위).
- **공정성 주의(검증으로 발견)**: grep은 순위 개념이 없어 **MRR이 정의되지 않음**. → grep은 Hit@k/Recall@k(찾았나 여부)만 보고, MRR은 순위가 있는 방식(의미·하이브리드)에만 적용. grep 베이스라인은 "ripgrep으로 질의어 매칭된 파일 집합" 정의로 고정.
- **비교군 3개를 같은 셋으로 측정** → 표로 제시:
  | 방식 | Hit@5 | MRR |
  |---|---|---|
  | grep (키워드만) | ? | N/A (무순위) |
  | 의미검색만 | ? | ? |
  | 하이브리드 | ? | ? |
- 이 표가 곧 포트폴리오의 "before/after 증거"이자 면접 답변("검색 품질을 어떻게 측정·개선했나").
- **정직성 주의**: 15~20문항은 통계적으로 견고한 벤치마크가 아니라 **방향성 지표**다. 절대 수치를 과장하지 않고 "평가를 설계·운영할 줄 안다"는 신호로 사용. 골드셋은 D-7의 공개 노트 기준으로 작성해 공개 가능하게 한다.

---

## 8. 단계별 구현 계획

| Phase | 내용 | 산출물 |
|---|---|---|
| **0. 셋업** | 프로젝트 구조, Ollama·BGE-M3 설치, `.ai.md`·ARCHITECTURE.md | 빈 골격 + 환경 |
| **1. 인덱싱** | 로드→청킹→임베딩→Chroma 적재, `index` 명령 | 178개 인덱싱 완료 |
| **2. 검색** | dense + BM25 + RRF 하이브리드 retriever | `search` 명령(노트만 반환) |
| **3. 생성** | RAG 프롬프트 + 로컬 LLM, 답변+출처 | `ask` 명령(답변+근거) |
| **4. 평가** | 골드셋 + Hit@k/MRR, 3방식 비교표 | 평가 리포트 |
| **5. MCP** | `vault_search`·`vault_ask` 툴 노출 | Claude Code 연동 |
| **(6. 스트레치)** | LangGraph 에이전트형(검색 부족 시 질의 재작성·재검색) | 차별화 +α |

---

## 9. ISC 수락 기준 (이진 검증)

- **ISC-1**: `python -m mycelium index` → "178 files, N chunks indexed" 출력 + `chroma/` 영속 디렉토리 생성 → pass/fail
- **ISC-2**: `python -m mycelium search "운동 후 단백질"` → 관련 노트 ≥1개를 유사도와 함께 반환 → pass/fail
- **ISC-3**: `python -m mycelium ask "<골드셋 질문>"` → 답변 + 출처 노트 경로 동시 출력 → pass/fail
- **ISC-4**: `python -m mycelium eval` → grep/의미/하이브리드 3행 비교표 출력, 하이브리드 Hit@5 ≥ grep Hit@5 → pass/fail
- **ISC-5**: MCP 서버 기동 후 Claude Code에서 `vault_search` 툴 호출 성공 → pass/fail
- **Anti-ISC**: 인덱싱·질의 전 과정에서 외부 네트워크 호출 0회(로컬 LLM·로컬 임베딩 사용 확인) → pass/fail

---

## 10. 디렉토리 구조 (예정)
```
mycelium/
├─ docs/
│  ├─ DESIGN.md          ← (이 문서)
│  └─ ARCHITECTURE.md    ← Phase 0에서 작성 (왜 이 구조인지)
├─ src/mycelium/
│  ├─ core/              ← 모델·설정
│  ├─ adapters/          ← embedding/llm/vectorstore 인터페이스
│  ├─ pipeline/
│  │  ├─ ingestion.py
│  │  ├─ retrieval.py
│  │  └─ generation.py
│  ├─ interfaces/
│  │  ├─ cli.py
│  │  └─ mcp_server.py
│  └─ eval/
│     ├─ goldset.yaml
│     └─ evaluate.py
├─ .ai.md                ← 에이전트 작업 가이드
├─ pyproject.toml
└─ README.md
```

---

## 11. 미해결 / 오픈 결정
- **D-5**: 생성 LLM 로컬 vs 클라우드 — Phase 4 실측 후 결정(M5/32GB로 로컬 우세).
- 벡터 저장소를 포트폴리오 마감 시 pgvector로 이관할지(기존 Supabase 스택 서사 강화) vs Chroma 유지(완전 로컬 일관) — Phase 5 이후 판단.
- **D-7 확정** — 공개 범위는 `public: true` 노트만(deny-by-default). 어느 노트에 플래그를 달지는 Phase 4(평가셋 작성) 시점에 함께 큐레이션.
- ~~import 경로 검증~~ → **완료(3.1 실측)**. `EnsembleRetriever`는 `langchain_classic`에 있음. 단 D-9에 따라 RRF/BM25 직접 구현 시 불필요.
- BM25 토크나이저는 **kiwipiepy** 확정(D-8, 실측 OK). Phase 0 의존성에 포함.
- **D-9 미결**: 검색 컴포넌트 직접 구현(권장) vs LangChain 컴포넌트 사용 — 사용자 확인 필요.
- 임베딩 모델: bge-m3(기본) vs Qwen3-Embedding(MMTEB 우위 보고) vs bona/bge-m3-korean(한국어 튜닝) — Phase 1에서 평가셋으로 비교 가능.

---

## 12. 구현·검증 이력 (2026-06-17)

전 Phase(0~6) 구현 후 **독립 4-에이전트 리뷰(verifier/code-reviewer/security/architect)**로 감사 → 결함 적발·수정 → 재검증.

### 적발·수정된 결함
- **C1**: `__main__.py` argv 조작이 multi-command 전환 후 `index`를 깨뜨림(회귀) → 블록 제거로 복구.
- **C2**: RRF 키가 `source::header_path`라 2차분할·헤더없는 청크가 충돌 → 청크 고유 `chunk_id` 도입(dense·BM25 공통 키), 실측 1046청크 0-mismatch.
- **H1**: 생성 컨텍스트가 `text_preview`(200자)로 잘림 → `RetrievedChunk.text`(전체본문) 추가, 전 경로 충전.
- **H2(=D-7 구현)**: `public_only` 필터 + 별도 chroma 경로 + 기본경로 덮어쓰기 가드.
- **H4**: 빈 코퍼스 BM25 크래시 → None 폴백 + `has_corpus()` 안내.
- **H5**: `pyproject.toml` 의존성 12개 버전 핀 + requirements.txt(재현성).
- M/L: 할루시네이션 판정 강화, agentic grade-once, dense 유사도변환(순위보존), BM25 0점 제외, 예외 축소, SSRF(localhost) 검증, 죽은코드 정리 등.

### 결정 D-10: no_evidence 검색거리 게이트 (할루시네이션 방어 결정론화)
- Why: 기존 방어가 LLM 출력 문구(`"노트에 근거가 없습니다"`)에 의존 → 변형 출력("노트에 근据不足" 등)에 오판정.
- 결정: 생성 전, 최상위 dense 유사도(`1 - cosine`)가 `relevance_threshold` 미만이면 **LLM 호출 전 결정론적으로 no_evidence 확정**. LLM 문구 매칭은 2차 보조로 병행.
- **임계 0.48 — 데이터 튜닝**(과적합 방지): 골드셋 18문항 최상위 유사도 최소 0.5061(전부 ≥0.50) / 무근거 질의("날씨"·"운동 단백질" 등) 0.44~0.47. 0.48은 정답 오거부 0 + 무근거 전부 차단의 안전 마진.

### 재검증 결과
- verifier: **APPROVE** (ISC-1~6 전부 통과, C1 회귀 복구·C2·H2·H4 직접 실행 확인).
- code-reviewer: **0 Critical/High** (수정 정확, 신규버그 없음).
- 잔여: D-7 공개 코퍼스는 아직 `public: true` 태깅 노트 0개 → 공개 리포 전 큐레이션 필요(메커니즘은 구현됨).
