# main.py

import json
from pathlib import Path

from rag.ingestion import (
    MarkdownDocumentLoader,
    IngestionPipeline,
    IngestionService,
)

from rag.rag_chunkers import HybridLegalChunker


class RAG:

    def __init__(self):

        base_path = Path(__file__).resolve()

        project_root = base_path.parents[1]

        rag_db_path = project_root / "rag_db"

        self.debug_path = project_root / "debug"

        loader = MarkdownDocumentLoader(str(rag_db_path))

        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(pipeline)

    def build_chunks(self):

        print("Loading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        self.save_chunks(chunks)

        return chunks

    def save_chunks(
            self,
            chunks,
            filename: str = "chunks.json"
    ):

        self.debug_path.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = self.debug_path / filename

        data = []

        for chunk in chunks:

            data.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "source": chunk.metadata.source,
                    "file": chunk.metadata.file,
                    "header": chunk.metadata.header,
                    "level": chunk.metadata.level,
                    "article_number": chunk.metadata.article_number,
                    "chunk_index": chunk.metadata.chunk_index,
                    "topics": chunk.metadata.topics
                }
            })

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(f"\n[DEBUG] Chunks saved:")
        print(output_file.resolve())

    def close(self):

        print("Shutting down...")


if __name__ == "__main__":

    rag = RAG()

    try:

        chunks = rag.build_chunks()

        print("\nDone.")

    finally:

        rag.close()