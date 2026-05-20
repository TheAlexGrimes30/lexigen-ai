from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from rag.dense_retriever import (
    Embedder,
    Retriever
)

from rag.reranker import (
    Reranker
)

from rag.index_service import (
    IndexService
)

from rag.ingestion import (
    MarkdownDocumentLoader,
    IngestionPipeline,
    IngestionService,
)

from rag.rag_chunkers import (
    HybridLegalChunker
)

from rag.retriever_dataset import (
    dataset
)

from rag.retriever_evaluation import (
    evaluate_rag
)

from rag.storage import (
    VectorStore
)


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

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            max_pool_size=50
        )

        self.reranker = Reranker(
            model_name="Qwen/Qwen3-Reranker-0.6B",
            top_n=5
        )

    def build_and_index(self):

        print("\nLoading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        self.index_if_needed(chunks)

        return chunks

    def index_if_needed(self, chunks):

        collection = self.vector_store.collection_name
        info = self.qdrant.get_collection(collection)

        points_count = info.points_count

        if points_count > 0:
            print(
                f"[Index] Skipping indexing — collection already has {points_count} points"
            )
            return

        print("[Index] Collection empty — starting indexing...")
        self.index_service.index(chunks)
        print("[Index] Done indexing")

    def search(
        self,
        query: str,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5,
        use_reranker: bool = True
    ):

        hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        if not use_reranker:
            return hits[:rerank_top_n]

        return self.reranker.rerank(
            query=query,
            hits=hits,
            top_n=rerank_top_n
        )

    def debug_search_pipeline(
        self,
        query: str,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5
    ):

        print("\n" + "=" * 100)
        print("[FULL SEARCH PIPELINE DEBUG]")
        print(f"QUERY: {query}")
        print("=" * 100)

        dense_hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        print("\n" + "=" * 100)
        print("RERANKER OUTPUT (FINAL RESULTS)")
        print("=" * 100)

        final_hits = self.reranker.rerank(
            query=query,
            hits=dense_hits,
            top_n=rerank_top_n
        )

        for i, hit in enumerate(final_hits, start=1):

            payload = hit.payload or {}

            print("\n" + "-" * 100)
            print(f"FINAL TOP {i}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"ARTICLE : {payload.get('article_number')}")
            print(f"HEADER  : {payload.get('header')}")
            print("\nTEXT:\n")
            print((hit.text or "")[:1200])

        print("\n" + "=" * 100)
        print("DENSE RETRIEVER OUTPUT (FOR COMPARISON)")
        print("=" * 100)

        for i, hit in enumerate(dense_hits, start=1):

            payload = hit.payload or {}

            print("\n" + "-" * 100)
            print(f"DENSE TOP {i}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"ARTICLE : {payload.get('article_number')}")
            print(f"HEADER  : {payload.get('header')}")

    def debug_dense_retrieval(self, query: str, top_k: int = 10):

        print("\n" + "=" * 100)
        print("[DENSE RETRIEVAL DEBUG]")
        print(f"QUERY: {query}")
        print("=" * 100)

        hits = self.retriever.retrieve(query=query, top_k=top_k)

        for i, hit in enumerate(hits, start=1):

            payload = hit.payload or {}

            print("\n" + "-" * 100)
            print(f"DENSE TOP {i}")
            print(f"SCORE   : {hit.score:.4f}")
            print(f"ARTICLE : {payload.get('article_number', 'unknown')}")
            print(f"HEADER  : {payload.get('header', 'unknown')}")
            print("\nTEXT:\n")
            print((hit.text or "")[:1200])

    def close(self):
        print("\nShutting down...")


if __name__ == "__main__":

    rag = RAG()

    try:

        chunks = rag.build_and_index()

        print("\nDone.\n")

        queries = [
            "что такое акцепт в гражданском праве",
            "что такое субсидиарная ответственность",
            "что такое солидарная ответственность",
            "что такое обязательство",
        ]

        print("\n" + "=" * 100)
        print("FULL SEARCH PIPELINE")
        print("=" * 100)

        for q in queries:
            rag.debug_search_pipeline(
                query=q,
                retrieve_top_k=20,
                rerank_top_n=5
            )

        print("\n" + "=" * 100)
        print("STARTING RERANK EVALUATION")
        print("=" * 100)

        evaluate_rag(
            rag=rag,
            dataset=dataset,
            output_path="rerank_eval.json",
            use_reranker=True,
            retrieve_top_k=20,
            rerank_top_n=5
        )

    finally:
        rag.close()