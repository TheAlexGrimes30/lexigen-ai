import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance

from backend.app.config import settings
from backend.modules.rag.chuncking import HybridLegalChunker
from backend.modules.rag.generator import ContextCleaner, CreditPromptBuilder, QwenClient, Generator
from backend.modules.rag.index_service import IndexService
from backend.modules.rag.ingestion_service import MarkdownDocumentLoader, IngestionPipeline, IngestionService
from backend.modules.rag.rag_embedder import Embedder
from backend.modules.rag.rag_service import RAGService, RAGMode
from backend.modules.rag.reranker_service import Reranker
from backend.modules.rag.retriever_service import Retriever
from backend.modules.rag.search_result_service import SearchResult
from backend.modules.rag.storage import VectorStore

load_dotenv()

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

    def __init__(self, n_ctx: int = 2048) -> None:
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
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )

        self.vector_store = VectorStore(
            client=self.qdrant,
            collection_name="credit_graph_collection",
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

        self.retriever.build_sparse_and_graph(chunks)
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
            query=query
        )

        return response.answer

    def analyze_contract(self, contract_text: str) -> str:
        response = self.rag_service.ask(
            query=contract_text,
            mode=RAGMode.DOCUMENT_ANALYSIS,
        )

        return response.answer


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
