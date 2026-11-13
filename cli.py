"""Terminal chat.  Run:  python cli.py"""
from rag import RAGEngine, cited_sources


def main() -> None:
    engine = RAGEngine()
    if engine.chunk_count() == 0:
        print("The index is empty. Run `python ingest.py` first.")
        return

    print("Ask about your documents. Type 'exit' to quit.\n")
    history: list[dict] = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in {"exit", "quit", ""}:
            if q:
                break
            continue

        retrieval = engine.retrieve(q, history)
        print("\nassistant> ", end="", flush=True)
        parts = []
        for token in engine.stream(retrieval):
            parts.append(token)
            print(token, end="", flush=True)
        answer = "".join(parts)
        print("\n")

        for s in cited_sources(answer, retrieval.sources):
            snippet = s.text[:160].replace("\n", " ")
            print(f"  [{s.index}] {s.filename}, p.{s.page}  ({s.score:.2f})  {snippet}…")
        print()

        history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]


if __name__ == "__main__":
    main()
