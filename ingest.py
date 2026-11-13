"""PDF ingestion: load -> clean -> chunk -> embed -> store in ChromaDB.

Ingestion is incremental and idempotent:
  * unchanged files (same SHA-256) are skipped
  * a changed file with the same name replaces its old chunks
  * chunk IDs are deterministic, so re-running never creates duplicates

CLI:
  python ingest.py                    # index every PDF in ./docs
  python ingest.py a.pdf some/dir     # index specific files / folders
  python ingest.py --force            # re-index even if unchanged
  python ingest.py --list             # show what's indexed
  python ingest.py --delete a.pdf     # remove a document from the index
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from rag import get_vectorstore

BATCH_SIZE = 128
MIN_CHUNK_CHARS = 30


@dataclass
class IngestResult:
    filename: str
    status: str  # indexed | skipped | empty | error
    chunks: int = 0
    pages: int = 0
    message: str = ""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join hyphenated line breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single newlines -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def find_pdfs(paths: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            found.extend(sorted(p.rglob("*.pdf")))
        elif p.suffix.lower() == ".pdf" and p.exists():
            found.append(p)
    return found


def ingest_pdf(path: Path, force: bool = False) -> IngestResult:
    vs = get_vectorstore()
    path = Path(path)
    name = path.name

    try:
        digest = file_sha256(path)

        if not force and vs.get(where={"source_hash": digest}, limit=1)["ids"]:
            return IngestResult(name, "skipped", message="already indexed, unchanged")

        pages = PyPDFLoader(str(path)).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks: list[Document] = []
        for page_doc in pages:
            text = clean_text(page_doc.page_content)
            if not text:
                continue
            page_no = int(page_doc.metadata.get("page", 0)) + 1  # 1-indexed for humans
            # Chunk per page so every chunk has an exact page number to cite.
            for piece in splitter.split_text(text):
                if len(piece) < MIN_CHUNK_CHARS:
                    continue
                chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "source": name,
                            "source_hash": digest,
                            "page": page_no,
                            "total_pages": len(pages),
                        },
                    )
                )

        if not chunks:
            return IngestResult(
                name,
                "empty",
                pages=len(pages),
                message="no extractable text (scanned PDF? run OCR first)",
            )

        ids = []
        for i, doc in enumerate(chunks):
            doc.metadata["chunk_index"] = i
            ids.append(f"{digest[:16]}:{i}")

        # Replace any older version of this file.
        vs._collection.delete(where={"source": name})

        for start in range(0, len(chunks), BATCH_SIZE):
            vs.add_documents(
                documents=chunks[start : start + BATCH_SIZE],
                ids=ids[start : start + BATCH_SIZE],
            )

        return IngestResult(name, "indexed", chunks=len(chunks), pages=len(pages))

    except Exception as e:  # corrupt/encrypted PDFs, etc.
        return IngestResult(name, "error", message=f"{type(e).__name__}: {e}")


def ingest_paths(
    paths: Iterable[Path],
    force: bool = False,
    progress: Callable[[int, int, IngestResult], None] | None = None,
) -> list[IngestResult]:
    pdfs = find_pdfs(paths)
    results = []
    for i, pdf in enumerate(pdfs, start=1):
        res = ingest_pdf(pdf, force=force)
        results.append(res)
        if progress:
            progress(i, len(pdfs), res)
    return results


def list_documents() -> list[dict]:
    """Summarize what's indexed: [{'source', 'chunks', 'pages'}], sorted by name."""
    data = get_vectorstore().get(include=["metadatas"])
    counts: dict[str, int] = defaultdict(int)
    pages: dict[str, int] = {}
    for m in data["metadatas"]:
        counts[m["source"]] += 1
        pages[m["source"]] = m.get("total_pages", 0)
    return [
        {"source": s, "chunks": counts[s], "pages": pages[s]} for s in sorted(counts)
    ]


def delete_document(filename: str) -> None:
    get_vectorstore()._collection.delete(where={"source": filename})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Index PDFs into the knowledge base.")
    ap.add_argument("paths", nargs="*", type=Path, help="PDF files or folders (default: ./docs)")
    ap.add_argument("--force", action="store_true", help="re-index even if unchanged")
    ap.add_argument("--list", action="store_true", help="list indexed documents and exit")
    ap.add_argument("--delete", metavar="FILENAME", help="remove a document from the index")
    args = ap.parse_args()

    if args.list:
        docs = list_documents()
        if not docs:
            print("Index is empty.")
        for d in docs:
            print(f"{d['source']}  ({d['pages']} pages, {d['chunks']} chunks)")
        return 0

    if args.delete:
        delete_document(args.delete)
        print(f"Deleted {args.delete}")
        return 0

    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)
    targets = args.paths or [config.DOCS_DIR]

    def show(i: int, n: int, r: IngestResult) -> None:
        detail = f"{r.chunks} chunks from {r.pages} pages" if r.status == "indexed" else r.message
        print(f"[{i}/{n}] {r.status:<8} {r.filename}  {detail}")

    results = ingest_paths(targets, force=args.force, progress=show)
    if not results:
        print(f"No PDFs found in: {', '.join(map(str, targets))}")
        return 1

    indexed = sum(r.status == "indexed" for r in results)
    print(f"\nDone: {indexed} indexed, {sum(r.status == 'skipped' for r in results)} skipped, "
          f"{sum(r.status in ('error', 'empty') for r in results)} problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
