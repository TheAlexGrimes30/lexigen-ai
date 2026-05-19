# main.py

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from rag.dense_retriever import Embedder, Retriever
from rag.index_service import IndexService
from rag.ingestion import (
    MarkdownDocumentLoader,
    IngestionPipeline,
    IngestionService,
)

from rag.rag_chunkers import HybridLegalChunker
from rag.retriever_dataset import dataset
from rag.retriever_evaluation import evaluate_rag
from rag.storage import VectorStore



class RAG:

    def __init__(self):

        base_path = Path(__file__).resolve()
        project_root = base_path.parents[1]

        rag_db_path = project_root / "rag_db"

        self.debug_path = (
            project_root / "debug"
        )

        loader = MarkdownDocumentLoader(
            str(rag_db_path)
        )

        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(
            pipeline
        )

        self.embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B",
            normalize=True
        )

        self.qdrant = QdrantClient(
            "localhost",
            port=6333
        )

        self.vector_store = VectorStore(
            client=self.qdrant,
            collection_name="credit_collection",
            vector_size=self.embedder.dim,
            distance=Distance.COSINE
        )

        self.vector_store.ensure_collection()

        self.index_service = IndexService(
            vector_store=self.vector_store,
            embedder=self.embedder
        )

        # FIXED
        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            max_pool_size=20
        )

    def build_and_index(self):

        print("\nLoading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        # self.save_chunks(chunks)

        self.index_if_needed(chunks)

        return chunks

    def index_if_needed(self, chunks):

        collection = (
            self.vector_store.collection_name
        )

        info = self.qdrant.get_collection(
            collection
        )

        points_count = info.points_count

        if points_count > 0:

            print(
                f"[Index] "
                f"Skipping indexing — "
                f"collection already has "
                f"{points_count} points"
            )

            return

        print(
            "[Index] Collection empty — "
            "starting indexing..."
        )

        self.index_service.index(chunks)

        print("[Index] Done indexing")

    def save_chunks(
        self,
        chunks,
        filename: str = "chunks.json"
    ):

        self.debug_path.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            self.debug_path / filename
        )

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
                    "article_number": (
                        chunk.metadata.article_number
                    ),
                    "chunk_index": (
                        chunk.metadata.chunk_index
                    ),
                    "topics": (
                        chunk.metadata.topics
                    )
                }
            })

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"\n[DEBUG] "
            f"Chunks saved: "
            f"{output_file.resolve()}"
        )

    def debug_retriever(
        self,
        query: str,
        top_k: int = 5
    ):

        print("\n" + "=" * 80)

        print(f"QUERY: {query}")

        hits = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )

        if not hits:

            print("No results")

            return

        for i, hit in enumerate(
            hits,
            start=1
        ):

            article = (
                hit.payload.get("article_number")
                if hit.payload
                else None
            )

            print("\n" + "-" * 80)

            print(f"TOP {i}")

            print(f"ARTICLE: {article}")

            print(
                f"SCORE: "
                f"{hit.score:.4f}"
            )

            print("\nTEXT:\n")

            print(
                (hit.text or "")[:1000]
            )

    def close(self):

        print("\nShutting down...")


if __name__ == "__main__":

    rag = RAG()

    try:

        chunks = rag.build_and_index()

        print("\nDone.\n")


        rag.debug_retriever(
            query="что такое акцепт в гражданском праве",
            top_k=5
        )

        rag.debug_retriever(
            query="что такое субсидиарная ответственность",
            top_k=5
        )

        rag.debug_retriever(
            query="что такое солидарная ответственность",
            top_k=5
        )

        rag.debug_retriever(
            query="что такое обязательство",
            top_k=5
        )


        print("\n" + "=" * 80)

        print(
            "STARTING "
            "RETRIEVER EVALUATION"
        )

        print("=" * 80)

        evaluate_rag(
            rag=rag,
            dataset=dataset,
            output_path="rag_eval_results.json"
        )

    finally:

        rag.close()