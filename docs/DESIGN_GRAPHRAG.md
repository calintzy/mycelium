# mycelium — Phase 7 설계: 그래프 증강 통합 검색 + Wiki 컴파운딩

> v1 하이브리드 검색을 **구조적으로 확장**한다. 그래프를 별도 모드가 아니라 **RRF의 세 번째 신호**로 융합하고, 커뮤니티 요약을 **검색 가능한 단위**로 인덱스에 넣어, `ask` 하나가 그래프·주제까지 인지하게 한다.
> 단일 진실: 본 문서 + `docs/DESIGN.md`(v1). 작성 2026-06-17 (통합형으로 재설계).

---

## 1. 목표 / 비목표

### 가치 명제 (정직 — 실측 기반)
이 볼트(180노트·v1 검색 이미 Hit@5 100%·그래프 절반)에서 GraphRAG의 가치는 **"검색 품질 향상"이 아니다**. 솔직히:
- (a) 링크/엔티티로 연결된 지식의 **멀티홉 추론**, (b) **주제·조망 합성**(커뮤니티), (c) **GraphRAG 직접 구축 = 면접·포트폴리오 역량 증명**.
- 검색 Hit@k 극적 개선을 약속하지 않는다(평가에서 들통남). 새 **능력**과 **기법 증명**이 본질.

### 목표
- **하나의 `ask`**가 dense + BM25 + **그래프 근접** + **커뮤니티 요약**을 융합(라우터·별도 모드 없음).
- 연결된 영역에서 멀티홉·주제 질문을 **기본 경로에서** 처리.
- 전수 조망은 **`--exhaustive` 파워모드**로만.
- **distill**로 좋은 Q&A를 위키 노트로 누적(컴파운딩).
- **생성 산출물(graph_report.md·graph.json 등) 인덱싱·그래프에서 제외**(노이즈 차단).

### 비목표
- Microsoft `graphrag` 라이브러리·Graphify 산출물 사용 안 함(직접 구축, 독립성). networkx/igraph/leidenalg만 커모디티.
- 별도 `graphask` 명령·flat/local/global 라우터 (← 추가형 폐기). 그래프는 `ask`에 통합.
- 실시간 그래프 갱신, 웹 그래프 시각화(후순위).

---

## 2. 확정 결정 (사용자)
| 항목 | 결정 |
|---|---|
| 통합 방식 | **그래프 근접·커뮤니티 요약을 RRF 신호로 융합** (별도 모드 X) (D-13) |
| 엣지 소스 | 위키링크+frontmatter 백본 + LLM 엔티티·관계 보강 (D-11) |
| 커뮤니티 | Leiden(leidenalg+igraph), 계층적 (D-12) |
| 전수 조망 | `--exhaustive` 파워모드로만 (D-17) |
| Wiki 컴파운딩 | distill 포함 (D-15) |
| 구축 철학 | 핵심 직접, 배관만 커모디티(D-9 연장) |

### 그라운딩 (2026-06-17 실측 — 추측 아님)
- GraphRAG(MS): 엔티티추출→그래프→Leiden 계층 커뮤니티→커뮤니티 요약→local/global.
- 의존성 실증: networkx 3.6.1 / igraph 1.0.0 / leidenalg OK, 미니 Leiden 동작.
- **백본 그래프 실측(`graph_probe.py`)**: 180노트, 링크 해소 401/405(98%), 엣지 266.
  - ⚠️ **고립 노드 45%(82개)** — 볼트 절반이 무링크(주로 raw·projects). → 링크만으론 부족, **LLM 엔티티 보강 필수**.
  - ⚠️ **최상위 허브가 Graphify 산출물**(`graph_report.md` deg58·27) = 노이즈. → 생성 산출물 **제외 필수**.
  - ✅ 링크된 최대 컴포넌트(87노트): Leiden 5커뮤니티 [28,28,16,9,6], **modularity 0.511**(의미 있는 구조).
- **정직한 함의**: 그래프는 주로 링크된 wiki 영역에서 작동. v1이 이미 골드셋 Hit@5 100% → 그래프의 검색 Hit@k 개선은 미미할 수 있음. 가치는 §1 참조.

### 이 통합형이 가능한 이유
- **RRF(D-9)가 이미 dense+BM25를 융합** → 신호 추가가 자연스러움.
- **C2의 chunk_id 키 통일** → 그래프 근접(노트단위)을 chunk_id로 전파해 같은 RRF에 합류 가능.
- **거리게이트(D-10)·가중치 데이터튜닝(D-3)** 그대로 재사용 → 노이즈 통제.

---

## 3. 아키텍처 (v1 retrieval을 확장, 신규 사일로 없음)
```
core/        ← GraphNode/Edge/Community 모델, graph·summary 설정
adapters/    ← (재사용) embedding/llm + graph_store(networkx 영속)
pipeline/
  retrieval.py   ← ★확장: RRF 입력에 graph_proximity + community_summary 추가
  generation.py  ← ★확장: 혼합 granularity(청크+요약) 컨텍스트 처리
  graph/                  ← Phase 7 신규
    build.py      ← 노드 + 위키링크/frontmatter 백본 + LLM 엔티티·관계
    community.py  ← Leiden 계층 커뮤니티
    summarize.py  ← 커뮤니티 LLM 요약(인덱스 시점·캐시) + 요약 임베딩 적재
    proximity.py  ← 시드 노트→그래프 이웃 확장, 노트점수→chunk_id 전파
    exhaustive.py ← (파워모드) 전 커뮤니티 map-reduce
  distill.py      ← Q&A→위키노트 write-back
interfaces/
  cli.py     ← ask(통합·기본) + ask --exhaustive + graph-build + distill
  mcp_server.py ← vault_ask가 자동으로 그래프 증강(통합)
```
> `graphask`·router 파일 없음. 그래프는 retrieval.py의 신호로 흡수. 의존방향(interfaces→pipeline→adapters→core) 유지.

---

## 4. 데이터 모델
- **NoteNode**(id=source), **EntityNode**(LLM 보강, id=정규화명).
- 엣지(타입드): `links_to`(위키링크), `related`/`tagged`(frontmatter), `mentions`(노트→엔티티), `relates_to`(엔티티↔엔티티, 라벨).
- **커뮤니티 요약 = 특수 검색 단위**: 청크와 동일하게 임베딩되어 벡터스토어에 저장(메타 `kind: "community_summary"`, community_id, 레벨). chunk와 한 인덱스에 공존.
- 지속화: `graph/graph.gpickle` + `graph/communities.json` + 요약은 벡터스토어 내. **프라이버시**: `graph/`는 원문 파생 → gitignore. 공개 리포는 sample_vault 재생성.

---

## 5. 인덱싱 (`graph-build`, 기존 `index` 후속/통합)
1. **백본 그래프(LLM 0)**: 노트→노드, `[[ ]]` 해소→`links_to`, frontmatter related/공유tags→엣지.
2. **LLM 엔티티·관계 보강(노트단위·해시캐시)**: 엔티티·트리플 추출 → 엔티티 노드/엣지. 정규화·병합. 179노트 ≈ 179콜(1회).
3. **Leiden 계층 커뮤니티** + **커뮤니티 LLM 요약(캐시)**.
4. **요약 임베딩 적재**: 각 커뮤니티 요약을 벡터스토어에 `kind:community_summary`로 추가(청크와 함께 검색됨).
5. **그래프 인접 영속**: proximity 계산용 adjacency 저장.

---

## 6. 질의 (`ask` — 통합·기본)
하나의 경로, RRF 다중 신호:
1. query → **dense + BM25**(청크) top-k (기존).
2. **graph_proximity**: 시드 노트(위 결과)의 그래프 이웃(N-홉, 엣지가중·관계타입) → 이웃 노트의 청크에 근접점수. 노트점수→chunk_id 전파.
3. **community_summary**: query 임베딩으로 요약 단위 검색(주제 질문이면 자연히 상위).
4. **RRF 융합**: dense·BM25·graph_proximity·summary 4신호를 chunk_id/summary_id 키로 융합. 가중치는 골드셋 데이터튜닝(D-3 방식, 그래프·요약 신호 비중 결정).
5. 거리게이트(D-10) → 혼합 결과(청크+요약)로 **답변 생성 + 근거(노트·경로·관계·커뮤니티)**.

### 파워모드 `ask --exhaustive`
- 전 커뮤니티 요약을 **map**(부분답변) → **reduce**(종합). 빠짐없는 전체 조망용. 느림·LLM 다수 — 명시 호출만.

---

## 7. Wiki 컴파운딩 (`distill`)
- `distill "<주제/직전 Q&A>"` → 큐레이션 위키노트(frontmatter `type:wiki`,`public:false`, 헤더 + 기존 노드에서 고른 `[[관련]]` 링크).
- 대상 `wiki_dir`(config). 동일 주제 기존 노트 있으면 업데이트(중복방지), 자동 덮어쓰기 금지·diff 미리보기.
- 작성 후 **증분 재인덱싱 + 그래프 갱신**(새 노드·엣지·요약 영향). 쓸수록 검색·그래프가 좋아지는 루프.

---

## 8. 평가
- 골드셋에 **그래프 수혜 문항** 추가: 멀티홉("A·B를 잇는 개념"), 주제("이 볼트 주요 테마").
- 비교: **평면 RRF(2신호) vs 통합 RRF(4신호)** 를 같은 골드셋에서 — Hit@k/MRR.
- **정직한 기대치(억측 금지)**: v1이 이미 Hit@5 100%라 기존 골드셋에선 **통합이 거의 동률일 것**. 그래프의 진짜 능력은 새 **멀티홉·주제 문항**으로 별도 측정. 그래도 검색 수치가 안 오르면 "이 코퍼스에선 검색 향상보다 능력 추가"로 정직 보고. **결과를 유리하게 조작 금지.**
- 가중치 튜닝: 그래프·요약 신호 비중을 골드셋으로 결정(단순 사실 노이즈 통제). 표본 한계 명시.

---

## 9. 설계 결정

### D-13(개정): 그래프·요약을 RRF 신호로 통합 (별도 모드 폐기)
- Why: 모든 질의가 그래프 혜택, 라우터 제거(브리틀), 명령 surface 축소, RRF·chunk_id·거리게이트 재사용 → 더 단순하면서 더 좋은 결과.
- Alt: 추가형(graphask 3모드+라우터) — 파편화·저활용·라우터 취약. 폐기.
- Threat: 단순 사실 질의에 그래프·요약 노이즈.
- Mitigation: RRF 가중치 골드셋 튜닝(그래프·요약 낮은 비중), 거리게이트.
- Open: 그래프 신호를 청크에 전파할 때 노트→청크 점수 분배식(균등 vs 대표청크) — 7.4 실측.

### D-11(개정): 엣지 = 링크 백본 + **전체 노트 LLM 보강(필수)**
- Why: 실측상 45%가 무링크 → 링크만으론 그래프가 볼트 절반만 덮음. 전체 180노트에 LLM 엔티티·관계 추출해 고립 노트를 공유개념으로 연결(전체 커버리지).
- 비용: 노트당 1콜 ≈ 180(1회)·해시캐시 후 0. **실측 16.5초/노트 → ~49분**(1회, 캐시 후 0).
- Threat: 추출 노이즈(엔티티 중복·환각 관계).
- Mitigation: 엔티티 정규화·병합, 관계는 근거 노트와 저장(검증가능). 구조화 출력(JSON).
- **생성 산출물 제외**: graph_report.md·graph.json 등 Graphify/자동생성 파일은 그래프·인덱싱에서 제외(실측상 deg58 노이즈 허브).

### D-12: Leiden 커뮤니티 (실증됨)
- networkx→igraph→leidenalg 계층 커뮤니티. 설치·동작 7.0 실증 완료. 실패시 networkx greedy 폴백.

### D-17: 전수 조망은 --exhaustive 파워모드
- Why: 통합형 기본(관련 요약 검색)이 빠르고 대다수 주제질문 충분. 전수 map-reduce는 느려 기본 부적합.
- Mitigation: 명시 플래그로만. 기본은 통합 경로.

### D-14: 커뮤니티 요약 인덱스 시점·캐시 / D-15: distill 컴파운딩(deny-by-default, 덮어쓰기 금지) — 이전과 동일.

### D-16(개정): 그래프는 통합(가산 아님). 그래프 미빌드 시 `ask`는 평면 2신호로 graceful 동작.

---

## 10. ISC (이진 검증)
- **ISC-7**: `graph-build` → "N nodes, M edges, K communities" + `graph/` 산출물 → pass/fail
- **ISC-8**: 백본만으로 엣지 ≥ (실측 위키링크 수 근접), 허브 노드 degree 합리적 → pass/fail
- **ISC-9**: 멀티홉 질문에서 `ask`(통합) 결과·근거에 **2홉 떨어진 노트가 그래프 신호로 포함**(평면 단독으론 누락) → pass/fail
- **ISC-10**: 주제 질문에서 `ask` 결과에 **커뮤니티 요약 단위가 상위 포함** → pass/fail
- **ISC-11**: `ask --exhaustive "<전체 주제>"` → 다수 커뮤니티 종합 답변 → pass/fail
- **ISC-12**: `distill "<주제>"` → 위키노트(frontmatter·[[링크]]) 생성 + 증분 재인덱싱·그래프 갱신 → pass/fail
- **ISC-13**: 그래프 골드셋에서 **통합 RRF(4신호) Hit@k ≥ 평면 RRF(2신호)** (아니면 정직 보고 + 가중치 튜닝) → pass/fail
- **Anti-ISC**: 외부 네트워크 0(로컬 LLM·로컬 그래프). graphrag·Graphify import 0.

---

## 11. 단계별 구현
| Phase | 내용 |
|---|---|
| 7.0 ✅ | 의존성 실증(networkx/igraph/leidenalg) |
| 7.1 | 백본 그래프(노드+위키링크/frontmatter), `graph-build` |
| 7.2 | LLM 엔티티·관계 보강(노트단위·캐시) |
| 7.3 | Leiden 계층 커뮤니티 + 요약(캐시) + 요약 임베딩 적재 |
| 7.4 | graph_proximity를 RRF 신호로(노트→chunk_id 전파) — retrieval.py 확장 |
| 7.5 | community_summary를 RRF 검색단위로 — `ask` 통합 |
| 7.6 | `--exhaustive` map-reduce 파워모드 |
| 7.7 | distill 컴파운딩 |
| 7.8 | 평가(평면 vs 통합 비교) + 가중치 데이터튜닝 |

---

## 12. 비용·성능 (정직)
| 단계 | 비용 |
|---|---|
| 백본 그래프 | 즉시(LLM 0) |
| LLM 엔티티 보강 | **실측 16.5초/노트 → 180노트 ≈ 49분**(1회·캐시 후 0). ⚠️"수 분"은 오류. 관계 라벨 노이즈(엔티티·mention 양호) |
| 커뮤니티 요약 | 커뮤니티 수콜·인덱스 1회·캐시 |
| `ask`(통합) | dense+BM25+그래프확장(로컬 계산)+요약검색 + 생성 1콜 — 평면과 비슷 |
| `ask --exhaustive` | 커뮤니티 수 map + reduce — 느림(명시 호출만) |
| distill | 1~2콜 + 증분 인덱싱 |

## 13. 독립성·프라이버시
- 위키링크 없는 볼트 → LLM-only 폴백("어떤 마크다운에도" 유지).
- 커모디티 의존만(graphrag·Graphify 미사용, D-9 연장).
- `graph/`·요약(원문 파생) → gitignore. 공개 리포 sample_vault 재생성.

## 14. 오픈
- 노트→청크 점수 전파식(7.4), 엔티티 정규화 강도(7.2), Leiden 해상도(7.3), 통합 RRF 가중치(7.8 튜닝).
