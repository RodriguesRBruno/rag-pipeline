"""Entry point: builds pipeline artifacts if missing, then runs an interactive Q&A loop."""

from __future__ import annotations

from src.pipeline import RAGPipeline, build_pipeline


def qa_loop(pipeline: RAGPipeline) -> None:
    print("\nRAG pipeline ready. Ask a question about Enter the Gungeon (or type 'quit' to exit).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break

        response = pipeline.query(question)
        print(f"\n{response.answer}\n")
        if response.sources:
            print("Sources:")
            for source in response.sources:
                print(f"  - {source.url} (document #{source.document_index})")
        print(f"[answerable={response.is_answerable}, confidence={response.confidence:.2f}]\n")


def main() -> None:
    pipeline = build_pipeline()
    qa_loop(pipeline)


if __name__ == "__main__":
    main()
