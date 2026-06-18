# mycelium MCP 서버 등록 가이드

mycelium MCP 서버를 Claude Code에 등록하면, Claude Code가 마크다운 볼트를 직접
의미검색하거나 RAG 답변을 생성할 수 있다. 기본 볼트는 동봉된 `sample_vault/`이며,
자신의 노트 폴더를 쓰려면 `VAULT_PATH` 환경변수로 지정한다.

---

## ⚠️ 프라이버시 경고 (D-7)

자신의 볼트를 인덱싱한 인덱스(`chroma/`)는 **절대 커밋 금지**다. git 사용 시 `git init` 후
`git check-ignore -v chroma/`로 .gitignore 제외를 반드시 검증하라(출력이 있으면 제외됨).
공개 리포에는 동봉된 합성 코퍼스 `sample_vault/`만 포함된다.

---

## 전제 조건

1. Ollama가 실행 중이어야 한다: `ollama serve`
2. 인덱싱이 완료돼 있어야 한다 (기본 볼트는 `sample_vault/`):
   ```bash
   cd /path/to/mycelium
   .venv/bin/python -m mycelium index
   ```

---

## Claude Code에 MCP 서버 등록

아래 명령을 터미널에서 실행한다.

```bash
claude mcp add mycelium -- /path/to/mycelium/.venv/bin/python -m mycelium serve
```

등록 확인:

```bash
claude mcp list
```

출력 예시:
```
mycelium: /path/to/mycelium/.venv/bin/python -m mycelium serve
```

---

## 사용 가능한 툴

등록 후 Claude Code 세션에서 아래 두 툴이 자동으로 인식된다.

### `vault_search`

볼트를 하이브리드 검색(dense 의미검색 + BM25 키워드 + RRF 융합)으로 검색한다.
단순히 관련 노트를 찾고 싶을 때 사용.

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | string | (필수) | 검색 질의 (한국어/영어) |
| `k` | integer | 5 | 반환할 결과 수 (1~50) |

반환값 예시 (sample_vault 기준):
```json
{
  "query": "하이브리드 검색에서 RRF로 두 신호를 합치는 이유",
  "k": 3,
  "total": 3,
  "results": [
    {
      "rank": 1,
      "source": "hybrid_search_notes.md",
      "header_path": "하이브리드 검색 메모 > RRF로 합치기",
      "preview": "Reciprocal Rank Fusion은 두 순위 리스트를 점수가 아니라 순위로 합친다...",
      "rrf_score": 0.027869,
      "dense_rank": 1,
      "bm25_rank": 2
    }
  ]
}
```

### `vault_ask`

볼트를 근거로 RAG 답변을 생성한다. 질문에 대한 자연어 답변과 근거 노트 목록을 함께 반환한다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | string | (필수) | 질문 (한국어/영어) |
| `k` | integer | 5 | 검색에 사용할 청크 수 (1~50) |

반환값 예시 (sample_vault 기준):
```json
{
  "query": "스테이크를 구운 뒤 왜 레스팅을 하나?",
  "answer": "구운 뒤 5~10분 쉬게 두면 육즙이 고기 전체로 재분배되어 퍽퍽함을 막습니다...",
  "no_evidence": false,
  "sources": [
    {
      "rank": 1,
      "source": "perfect_steak_guide.md",
      "rrf_score": 0.027869
    }
  ]
}
```

---

## 성능 참고

- **첫 번째 툴 호출**: HybridRetriever 초기화(Chroma 로드 + BM25 인덱스 구축)로 10~20초 소요.
- **이후 호출**: 싱글턴 재사용으로 즉시 응답.
- LLM(vault_ask)은 첫 호출 시 Ollama 모델 로드로 추가 20~30초 소요 가능.

---

## 등록 해제

```bash
claude mcp remove mycelium
```

---

## 수동 실행 (디버그용)

```bash
cd /Users/ryan/ClaudeProject/mycelium
.venv/bin/python -m mycelium serve
```

stdio로 MCP 프로토콜 메시지를 주고받는 서버가 기동된다.
