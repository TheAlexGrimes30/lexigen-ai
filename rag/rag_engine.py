from pathlib import Path

from qdrant_client import QdrantClient

from rag.generator import QwenClient, Generator, ContextCleaner, CreditPromptBuilder
from rag.index_service import IndexService
from rag.ingestion import MarkdownDocumentLoader, IngestionPipeline
from rag.ingestion_service import IngestionService
from rag.rag_chuckers import SentenceChunker, StructureChunker, SmartChunker, WindowChunker
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
        print(rag_db_path)

        loader = MarkdownDocumentLoader(str(rag_db_path))

        sentence_chunker = SentenceChunker(
            chunk_size=800,
            overlap_sentences=2
        )

        window_chunker = WindowChunker(
            max_chars=800,
            overlap=150
        )

        structure_chunker = StructureChunker(
            fallback_chunker=sentence_chunker
        )

        chunker = SmartChunker(
            structure_chunker=structure_chunker,
            sentence_chunker=sentence_chunker,
            window_chunker=window_chunker,
            max_chars=800
        )

        pipeline = IngestionPipeline(
            loader=loader,
            chunker=chunker
        )

        self.ingestion = IngestionService(pipeline)

        embedder = Embedder(
            model_name="Qwen/Qwen3-Embedding-0.6B"
        )

        client = QdrantClient(host="localhost", port=6333)

        vector_store = VectorStore(
            client=client,
            collection_name="rag_credit_collection",
            vector_size=embedder.dim
        )

        vector_store.ensure_collection()

        self.index_service = IndexService(vector_store, embedder)

        retriever = Retriever(
            vector_store=vector_store,
            embedder=embedder
        )

        llm = QwenClient()

        generator = Generator(
            llm=llm,
            prompt_builder=CreditPromptBuilder(),
            cleaner=ContextCleaner()
        )

        reranker = Reranker()
        self.rag_service = RAGService(retriever, reranker, generator)
        self.client = client
        self.generator = generator

        print("Running ingestion...")
        chunks = self.ingestion.load_chunks()

        print(f"\n[DEBUG] Total chunks: {len(chunks)}")

        for c in chunks[:20]:
            print(c.metadata.article_number, "|", c.metadata.header)

        for i, c in enumerate(chunks[:5]):
            payload = c.to_payload()

            print(f"\n--- CHUNK {i} ---")
            print(f"text: {c.text[:200]}")
            print(f"file: {payload.get('file')}")
            print(f"article: {payload.get('article_number')}")
            print(f"header: {payload.get('header')}")

        print("Indexing...")
        self.index_service.index(chunks)


    def ask(self, query: str) -> RAGResponse:
        return self.rag_service.ask(query)

    def close(self):
        print("Shutting down RAG...")

        if hasattr(self.generator, "llm") and hasattr(self.generator.llm, "close"):
            self.generator.llm.close()

        self.client.close()


if __name__ == "__main__":

    rag = RAG()

    questions = [
        "какие действия может выполнять должник"
        "можно ли исполнять обязательства с использованием технологий",
        "кому должно исполняться обязательство"
    ]

    for q in questions:
        print("\nQ:", q)
        res = rag.ask(q)
        print("A:", res.answer)
