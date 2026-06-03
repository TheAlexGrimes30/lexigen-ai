from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from backend.modules.rag.chuncking import HybridLegalChunker
from backend.modules.rag.dense_retriever_service import Retriever
from backend.modules.rag.generator import ContextCleaner, CreditPromptBuilder, QwenClient, Generator
from backend.modules.rag.index_service import IndexService
from backend.modules.rag.ingestion_service import MarkdownDocumentLoader, IngestionPipeline, IngestionService
from backend.modules.rag.rag_embedder import Embedder
from backend.modules.rag.rag_service import RAGService, RAGMode
from backend.modules.rag.reranker_service import Reranker
from backend.modules.rag.search_result_service import SearchResult
from backend.modules.rag.storage import VectorStore


class RAG:
    """
    Main RAG pipeline.

    Responsibilities:
    - document ingestion
    - chunk indexing
    - vector retrieval
    - reranking
    - answer generation
    """

    def __init__(self, n_ctx: int = 2048, max_tokens: int = 400) -> None:
        base_path = Path(__file__).resolve()
        project_root = base_path.parents[3]

        model_path = project_root / "models" / "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
        rag_db_path = project_root / "rag_db"

        loader = MarkdownDocumentLoader(str(rag_db_path))
        chunker = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker,
        )

        self.ingestion = IngestionService(pipeline)

        self.embedder = Embedder(
            model_name=str(project_root / "models" / "Qwen3-Embedding-0.6B"),
            normalize=True,
        )

        self.qdrant = QdrantClient(
            "localhost",
            port=6333,
        )

        self.vector_store = VectorStore(
            client=self.qdrant,
            collection_name="credit_collection",
            vector_size=self.embedder.dim,
            distance=Distance.COSINE,
        )

        self.vector_store.ensure_collection()

        self.index_service = IndexService(
            vector_store=self.vector_store,
            embedder=self.embedder,
        )

        self.retriever = Retriever(
            vector_store=self.vector_store,
            embedder=self.embedder,
            max_pool_size=50,
        )

        self.reranker = Reranker(
            model_name=str(project_root / "models" / "bge-reranker-v2-m3"),
            top_n=5,
        )

        self.llm = QwenClient(
            model_path=str(model_path),
            n_ctx=n_ctx,
        )

        self.generator = Generator(
            llm=self.llm,
            prompt_builder=CreditPromptBuilder(),
            cleaner=ContextCleaner(),
        )

        self.rag_service = RAGService(
            retriever=self.retriever,
            reranker=self.reranker,
            generator=self.generator,
            min_final_score=0.50,
        )

    def build_and_index(self) -> list:
        """
        Load chunks and index them if collection is empty.

        Returns:
            List:
                List of loaded chunks.
        """

        print("\nLoading chunks...\n")

        chunks = self.ingestion.load_chunks()

        print(f"Loaded chunks: {len(chunks)}")

        self.index_if_needed(chunks)

        return chunks

    def index_if_needed(
            self,
            chunks: list
    ) -> None:
        """
        Index chunks only if collection is empty.

        Args:
            chunks (List):
                List of document chunks.

        Returns:
            None
        """

        collection = self.vector_store.collection_name

        info = self.qdrant.get_collection(
            collection
        )

        points_count = info.points_count

        if points_count > 0:
            print(
                f"[Index] Skipping indexing — "
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

    def search(
            self,
            query: str,
            retrieve_top_k: int = 20,
            rerank_top_n: int = 5,
            use_reranker: bool = True
    ) -> list[SearchResult]:
        """
        Perform retrieval and optional reranking.

        Args:
            query (str):
                User query.

            retrieve_top_k (int):
                Number of retrieved chunks.

            rerank_top_n (int):
                Number of final reranked chunks.

            use_reranker (bool):
                Whether to apply reranking.

        Returns:
            List[SearchResult]:
                Final search results.
        """

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

    def ask(self, query: str) -> str:
        response = self.rag_service.ask(
            query=query,
            mode=RAGMode.USER_QUERY,
        )

        return response.answer

    def analyze_contract(self, contract_text: str) -> str:
        response = self.rag_service.ask(
            query=contract_text,
            mode=RAGMode.DOCUMENT_ANALYSIS,
        )

        return response.answer

    def debug_dense_retrieval(
            self,
            query: str,
            top_k: int = 10
    ) -> None:
        """
        Debug dense retrieval results.

        Args:
            query (str):
                Search query.

            top_k (int):
                Number of retrieved chunks.

        Returns:
            None
        """

        print("\n" + "=" * 100)

        print("[DENSE RETRIEVAL DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        hits = self.retriever.retrieve(
            query=query,
            top_k=top_k
        )

        for i, hit in enumerate(
                hits,
                start=1
        ):
            payload = hit.payload or {}

            print("\n" + "-" * 100)

            print(f"DENSE TOP {i}")

            print(
                f"SCORE   : "
                f"{hit.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{payload.get('article_number', 'unknown')}"
            )

            print(
                f"HEADER  : "
                f"{payload.get('header', 'unknown')}"
            )

            print("\nTEXT:\n")

            print(
                (hit.text or "")[:1200]
            )

    def debug_search_pipeline(
            self,
            query: str,
            retrieve_top_k: int = 20,
            rerank_top_n: int = 5
    ) -> None:
        """
        Debug full retrieval + reranking pipeline.

        Args:
            query (str):
                User query.

            retrieve_top_k (int):
                Retriever top-k.

            rerank_top_n (int):
                Final reranker top-n.

        Returns:
            None
        """

        print("\n" + "=" * 100)

        print("[FULL SEARCH PIPELINE DEBUG]")

        print(f"QUERY: {query}")

        print("=" * 100)

        dense_hits = self.retriever.retrieve(
            query=query,
            top_k=retrieve_top_k
        )

        print("\n" + "=" * 100)

        print("RERANKER OUTPUT")

        print("=" * 100)

        final_hits = self.reranker.rerank(
            query=query,
            hits=dense_hits,
            top_n=rerank_top_n
        )

        for i, hit in enumerate(
                final_hits,
                start=1
        ):
            payload = hit.payload or {}

            print("\n" + "-" * 100)

            print(f"FINAL TOP {i}")

            print(
                f"SCORE   : "
                f"{hit.score:.4f}"
            )

            print(
                f"ARTICLE : "
                f"{payload.get('article_number')}"
            )

            print(
                f"HEADER  : "
                f"{payload.get('header')}"
            )

            print("\nTEXT:\n")

            print(
                (hit.text or "")[:1200]
            )

    def close(self) -> None:
        """
        Release resources.

        Returns:
            None
        """

        try:
            self.generator.close()
        except Exception:
            pass

        print("\nShutting down...")
