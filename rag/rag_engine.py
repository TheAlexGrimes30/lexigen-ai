from pathlib import Path

from qdrant_client import QdrantClient

from rag.generator import (
    QwenClient,
    Generator,
    CreditPromptBuilder,
    ContextCleaner,
)

from rag.index_service import IndexService
from rag.ingestion import (
    MarkdownDocumentLoader,
    IngestionPipeline,
    IngestionService,
)

from rag.rag_chuckers import HybridLegalChunker
from rag.rag_config import RAGResponse
from rag.rag_service import RAGService
from rag.reranker import Reranker
from rag.retriever import Embedder, Retriever
from rag.storage import VectorStore


class RAG:

    def __init__(self):

        base_path = Path(__file__).resolve()
        project_root = base_path.parents[1]

        rag_db_path = project_root / "rag_db"

        loader = MarkdownDocumentLoader(str(rag_db_path))

        parser = HybridLegalChunker()

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=parser
        )

        self.ingestion = IngestionService(pipeline)

        embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        self.client = QdrantClient(
            host="localhost",
            port=6333
        )

        vector_store = VectorStore(
            client=self.client,
            collection_name="credit_collection",
            vector_size=embedder.dim
        )

        vector_store.ensure_collection()

        self.index_service = IndexService(
            vector_store=vector_store,
            embedder=embedder
        )

        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder
        )

        reranker = Reranker()

        llm = QwenClient(
            url="http://localhost:11434",
            model="qwen2.5:3b"
        )

        generator = Generator(
            llm=llm,
            prompt_builder=CreditPromptBuilder(),
            cleaner=ContextCleaner()
        )

        self.rag_service = RAGService(
            retriever=retriever,
            reranker=reranker,
            generator=generator
        )

        print("Running ingestion...")

        self.chunks = self.ingestion.load_chunks()

        print(f"\n[DEBUG] Total chunks: {len(self.chunks)}")

        for c in self.chunks[:10]:
            print(
                c.metadata.article_number,
                "|",
                c.metadata.header
            )

        print("\nIndexing...")

        self.index_service.index(self.chunks)

        print("RAG initialized.")

    def ask(self, query: str) -> RAGResponse:
        return self.rag_service.ask(query)

    def close(self):

        print("Shutting down RAG...")

        try:
            self.client.close()

        except Exception as e:
            print(
                "[WARN] Qdrant close error:",
                repr(e)
            )


if __name__ == "__main__":

    rag = RAG()

    questions = [
        "какие действия может выполнять должник"
    ]

    try:

        for q in questions:

            print("\nQ:", q)

            res = rag.ask(q)

            print("\nA:")
            print(res.answer)

    finally:
        rag.close()
