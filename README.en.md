# mycelium

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![LLM: local (Ollama)](https://img.shields.io/badge/LLM-local%20(Ollama)-success)
![RAG: hybrid + GraphRAG](https://img.shields.io/badge/RAG-hybrid%20%2B%20GraphRAG-orange)

**English** · [한국어](README.md)

> Mycelium is the vast underground network of fungal threads that connects plants and shares nutrients between them.
> This project does the same for scattered markdown notes — connecting them through search, a knowledge graph, and RAG, so knowledge reinforces itself and grows like a living network.

A **local-first hybrid search + RAG Q&A tool that works on any folder of markdown files.** Ask a question and a local LLM answers grounded in your notes, **with the source notes cited**. It does not depend on Obsidian — a directory of `.md` files is all you need.

The repo ships a fully synthetic corpus `sample_vault/` (15 public notes) so you can reproduce indexing, search, and evaluation right after cloning. To use your own notes, just set `VAULT_PATH`.

---

## Example

Using the bundled `sample_vault/` — the answer is found **by meaning**, even though the note never contains the exact query word "레스팅" (resting):

```console
$ myco ask "스테이크 레스팅을 왜 하나?"   # "Why rest a steak?"

구운 후 5~10분 동안 쉬게 하는 것은 육즙이 고기 전체로 재분배되어
더욱 촉촉한 식감을 주기 위함이다. 바로 자르면 육즙이 흘러나와 퍽퍽해진다.
# (Resting for 5–10 min lets the juices redistribute and keeps the meat moist;
#  cutting immediately lets them run out.)

근거 노트 (sources):
  [1] sous_vide_basics.md
  [2] perfect_steak_guide.md
```

---

## Features
- **Hybrid search** — semantic (dense, BGE-M3) + keyword (BM25), fused with RRF.
- **Real Korean support** — a kiwipiepy morphological tokenizer is injected into BM25, avoiding the whitespace-tokenization trap for Korean.
- **Relevance gate** — if the top dense similarity is below a threshold, it returns "no evidence" *before* calling the LLM (hallucination defense).
- **Fully local** — embedding and generation both run on Ollama. Nothing leaves your machine.
- **GraphRAG** — builds a graph from wikilinks + LLM-extracted entities, and folds graph proximity and community summaries into the same RRF ranking.
- **Distill compounding** — distills good Q&A into curated wiki notes, so the corpus improves the more you use it.
- **Retrieval evaluation** — compares grep vs semantic vs hybrid on the same gold set (Hit@k / MRR).
- **MCP server** — standard MCP (stdio), so Claude Code, Codex, and other MCP clients can search the vault directly.

---

## Architecture

Dependency direction: `interfaces → pipeline → adapters → core` (clean architecture).

```
interfaces/   CLI (typer) + MCP server (FastMCP) — I/O only
pipeline/     ingestion (load·chunk·embed·index) / retrieval (hybrid) /
              generation (RAG) / graph (backbone·communities·summaries) / distill
adapters/     embedding / llm / vectorstore / graph_store — local↔cloud swap point
core/         domain models·config — minimal external deps
```

Data flow: `.md → header-based chunking (+ secondary split of oversized sections) → BGE-M3 embedding → Chroma` + a BM25 index. At query time, dense·BM25 (+ graph·summary) ranks are fused with RRF → relevance gate → a local LLM generates a source-cited answer.

Single source of truth for design: [`docs/DESIGN.md`](docs/DESIGN.md) (D-1–D-10), [`docs/DESIGN_GRAPHRAG.md`](docs/DESIGN_GRAPHRAG.md) (Phase 7, D-11–D-17).

---

## Results

### Own-vault gold set (directional — small sample)

| Method | Hit@5 |
|---|---|
| grep (keyword only) | 77.8% |
| Hybrid (dense + kiwi BM25 + RRF) | 100% |

grep has no ranking, so MRR is undefined (DESIGN §7). A small gold set is a directional indicator, not a statistical benchmark — the numbers are not overstated.

### KorQuAD v1 independent benchmark (an unsaturated public Korean set)

On the own vault, dense alone is already at Hit@5 100%, so technique gains don't show. To break saturation, the same three methods were compared on a public Korean QA set (KorQuAD v1).

| Method | Hit@5 |
|---|---|
| dense only | 97.0% |
| dense + whitespace BM25 (off-the-shelf hybrid) | 96.8% |
| dense + kiwi BM25 (our hybrid) | **98.0%** |

Key finding: **whitespace BM25 is actually lower than dense (96.8%)** — splitting Korean on whitespace turns the keyword signal into noise. Only **with a kiwi morphological tokenizer (98.0%)** does the hybrid actually beat dense (D-8). Reproducible via `benchmarks/korquad_bench.py`.

> Honesty: results are reported as-is, favorable or not. On this corpus the graph signal does not raise retrieval Hit@k, so its RRF weight is set to 0 and it is used only for multi-hop / overview "capability" (DESIGN_GRAPHRAG §8).

---

## Quick start

### 1. Install
```bash
git clone https://github.com/calintzy/mycelium
cd mycelium
python3.12 -m venv .venv
.venv/bin/pip install -e .
```

### 2. Ollama + models
```bash
# after installing https://ollama.com
ollama serve
ollama pull bge-m3        # embedding (1.2GB)
ollama pull qwen2.5:14b   # generation LLM (9GB)
```

### 3. Index (default vault = bundled sample_vault/)
```bash
.venv/bin/python -m mycelium index
```

### 4. Search · Ask · Evaluate · Graph
```bash
.venv/bin/python -m mycelium search "RRF in hybrid search" --k 3
.venv/bin/python -m mycelium ask "Why rest a steak?"
.venv/bin/python -m mycelium eval          # grep vs semantic vs hybrid table
.venv/bin/python -m mycelium graph-build   # graph + communities + summaries (LLM calls)
```

`myco` works the same as `python -m mycelium` (with the editable install).

### Use your own notes
```bash
VAULT_PATH=~/MyNotes .venv/bin/python -m mycelium index
```
Indexes of your real vault (`chroma/`, `graph/`) are `.gitignore`d and never committed (D-7).

---

## Interfaces
- **CLI (primary)** — `index` / `search` / `ask` / `eval` / `graph-build` / `distill` / `agentic` / `serve`.
- **MCP (optional)** — standard MCP for Claude Code, Codex, etc. See [`docs/MCP_SETUP.md`](docs/MCP_SETUP.md).

---

## Stack
Python 3.12 · LangChain 1.3.x · Ollama (`bge-m3` + `qwen2.5:14b`) · Chroma · rank_bm25 + kiwipiepy · networkx + igraph + leidenalg · MCP

---

## Privacy (D-7, deny-by-default)
The public repo contains only the synthetic `sample_vault/`. Artifacts from indexing your own vault (`chroma/`, `graph/`) and your private gold set (`*.local.yaml`) are never committed. `index --public-only` indexes only notes with frontmatter `public: true`, into a separate path.

---

## License
MIT — see [`LICENSE`](LICENSE).
