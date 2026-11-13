# PDF Knowledge Base (LangChain + ChromaDB)

Ask questions about your PDFs (research papers, company docs) and get answers grounded in the text, with numbered citations that link back to the file and page.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add ANTHROPIC_API_KEY (or OPENAI_API_KEY)

# put PDFs in ./docs, then:
python ingest.py              # build the index
streamlit run app.py          # chat UI  (or: python cli.py)
```

You can also upload PDFs straight from the sidebar of the web UI.

Embeddings run locally by default (`BAAI/bge-small-en-v1.5`), so only the answering LLM needs an API key. The first run downloads the embedding model (~130 MB).

## How it works

```
PDFs ─► PyPDFLoader ─► clean ─► split per page (1000 chars, 200 overlap)
                                      │  metadata: source, page, sha256
                                      ▼
                        embeddings ─► ChromaDB (persisted, cosine)

question ─► [rewrite follow-ups into standalone question using chat history]
         ─► top-k similarity search
         ─► prompt: numbered excerpts [1]..[k] + "answer only from these, cite [n]"
         ─► streamed answer ─► parse [n] ─► show only the sources actually cited
```

| File | Role |
|---|---|
| `config.py` | All settings (env-overridable) |
| `ingest.py` | Load, clean, chunk, embed, store. Incremental and idempotent; also a CLI |
| `rag.py` | Vector store, retrieval, grounded prompt, streaming, citation parsing |
| `app.py` | Streamlit chat with upload, document list/delete, source panels |
| `cli.py` | Terminal chat |

## Design choices worth knowing

- **Grounding:** the prompt forbids outside knowledge and requires `[n]` citations. If retrieval returns nothing (see `MIN_RELEVANCE`), the LLM isn't called at all.
- **Citations are verified in code:** the UI shows only sources whose `[n]` actually appears in the answer. Tick "Also show passages the answer didn't cite" to inspect everything retrieved.
- **Page-accurate sources:** chunks never span pages, so every citation has an exact page. The cost is that a passage split across a page break is indexed as two chunks.
- **Incremental ingestion:** files are tracked by SHA-256. Unchanged files are skipped, edited files replace their old chunks, and `--force` re-indexes everything.
- **Follow-up questions:** "What about its limitations?" is rewritten into a standalone query before searching, which is what makes multi-turn chat retrieve correctly.
- **Switching embedding models** creates a new collection automatically (the model name is part of the collection name), so you re-run `python ingest.py` instead of corrupting the old index.
- **Prompt injection:** retrieved text is labelled untrusted in the prompt. This reduces, but doesn't eliminate, the risk if you index documents you don't control.

## Tuning

| Symptom | Try |
|---|---|
| Answers miss details that are in the docs | Raise `TOP_K` (6 to 10), or lower `CHUNK_SIZE` to 600 to 800 |
| Answers lack surrounding context | Raise `CHUNK_SIZE` (1200 to 1500) and `CHUNK_OVERLAP` |
| Irrelevant questions still get answers | Set `MIN_RELEVANCE` (start at 0.3; scores depend on the embedding model, so check the relevance values shown in the UI first) |
| Poor retrieval on technical or scientific text | Use a larger embedding model, e.g. `BAAI/bge-base-en-v1.5` or OpenAI `text-embedding-3-large` |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` only affects newly indexed files. Run `python ingest.py --force` to rebuild.

## Limitations

- **Scanned PDFs** have no text layer and are reported as "no extractable text". Run OCR first (e.g. `ocrmypdf in.pdf out.pdf`).
- **Tables, equations, and multi-column layouts** are flattened by `pypdf`. For table-heavy documents, swap the loader for `PyMuPDF4LLM` or `unstructured`; only `ingest_pdf` needs to change.
- Two files with the same filename in different folders are treated as the same document.
- Single-user and local: Chroma runs in-process. For multi-user deployment, run Chroma as a server and put auth in front of the UI.

## Ideas for next steps

Hybrid search (BM25 + vectors), a cross-encoder reranker over the top 20 results, metadata filters ("only search paper X"), and an offline eval set of question/expected-page pairs to measure changes to chunking or models.
