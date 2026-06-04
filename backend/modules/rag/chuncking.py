import hashlib
import re
from typing import Any

from chonkie import SentenceChunker
from chonkie.refinery import OverlapRefinery

from backend.modules.rag.rag_config import ChunkMetadata, Chunk


class Sectioner:
    """
    Markdown section parser.
    """

    def extract_sections(self, text: str) -> list[dict]:
        """
        Extract sections from a Markdown document.
        """

        sections = []
        current = {"header": None, "level": 0, "content": []}

        for line in text.split("\n"):
            line = line.rstrip()

            if not line.strip():
                continue

            match = re.match(r"^(#{1,6})\s+(.+)", line)

            if match:
                if current["content"]:
                    sections.append(current)

                current = {
                    "header": match.group(2).strip(),
                    "level": len(match.group(1)),
                    "content": [],
                }
                continue

            current["content"].append(line)

        if current["content"]:
            sections.append(current)

        return sections


class ContextInjector:
    """
    Injects legal context into chunk text.
    """

    def inject(
        self,
        *,
        article_number: str | None,
        header: str | None,
        article_title: str | None,
        legal_domain: str | None,
        topics: list[str],
        keywords: list[str],
        text: str,
    ) -> str:
        """
        Add structured context directly to text for dense retrieval and reranking.
        """

        context = []

        if article_number:
            context.append(f"Статья {article_number}")

        if article_title and article_title not in context:
            context.append(article_title)

        if header and header not in context:
            context.append(header)

        if legal_domain:
            context.append(f"Область: {legal_domain}")

        if topics:
            context.append("Темы: " + ", ".join(str(t) for t in topics[:8]))

        if keywords:
            context.append("Ключевые слова: " + ", ".join(str(k) for k in keywords[:10]))

        prefix = " | ".join(context)

        return f"[{prefix}]\n\n{text}" if prefix else text


class ChunkValidator:
    """
    Validates chunk quality.
    """

    def __init__(self, min_chars: int = 120, min_words: int = 20):
        self.min_chars = min_chars
        self.min_words = min_words

    def is_valid(self, text: str) -> bool:
        """
        Validate whether a chunk is useful for RAG.
        """

        text = text.strip()

        if len(text) < self.min_chars:
            return False

        if len(text.split()) < self.min_words:
            return False

        alpha_ratio = sum(ch.isalpha() for ch in text) / max(len(text), 1)

        return alpha_ratio >= 0.25


class FrontmatterAdapter:
    """
    Extracts domain-neutral RAG metadata from Markdown frontmatter.
    """

    @staticmethod
    def as_list(value: Any) -> list[str]:
        """
        Convert scalar/list metadata value to list[str].
        """

        if value is None:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        text = str(value).strip()

        return [text] if text else []

    @staticmethod
    def extract_article(header: str | None, frontmatter: dict) -> str | None:
        """
        Extract legal article number from header or metadata.
        """

        if header:
            match = re.search(r"Статья\s+(\d+(?:\.\d+)?)", header)
            if match:
                return match.group(1)

        for key in ("article_number", "article", "article_id", "norm_id"):
            value = frontmatter.get(key)
            if value:
                return str(value).strip()

        doc_id = str(frontmatter.get("id", ""))
        match = re.search(r"article_(\d+)(?:_(\d+))?", doc_id)

        if not match:
            return None

        if match.group(2):
            return f"{match.group(1)}.{match.group(2)}"

        return match.group(1)

    @classmethod
    def build_metadata(
        cls,
        *,
        frontmatter: dict,
        filepath: str,
        header: str | None,
        level: int | None,
        chunk_index: int,
    ) -> ChunkMetadata:
        """
        Build rich ChunkMetadata from Markdown frontmatter.
        """

        classic_rag = frontmatter.get("classic_rag") or {}
        tree_rag = frontmatter.get("tree_rag") or {}
        graph_rag = frontmatter.get("graph_rag") or {}

        topics = []
        topics.extend(cls.as_list(frontmatter.get("topics")))
        topics.extend(cls.as_list(classic_rag.get("topics")))

        keywords = []
        keywords.extend(cls.as_list(frontmatter.get("keywords")))
        keywords.extend(cls.as_list(classic_rag.get("keywords")))

        related_articles = []
        related_articles.extend(cls.as_list(frontmatter.get("related_articles")))

        if isinstance(graph_rag, dict):
            for relation in graph_rag.get("relations") or []:
                if not isinstance(relation, dict):
                    continue
                target = relation.get("target") or relation.get("to")
                if target:
                    related_articles.append(str(target))

        hierarchy = tree_rag.get("hierarchy") if isinstance(tree_rag, dict) else {}

        article_title = (
            frontmatter.get("title")
            or frontmatter.get("article_title")
            or header
        )

        return ChunkMetadata(
            source=frontmatter.get("source", "unknown"),
            file=filepath,
            header=header,
            level=level,
            article_number=cls.extract_article(header, frontmatter),
            chunk_index=chunk_index,
            topics=sorted(set(topics)),
            keywords=sorted(set(keywords)),
            related_articles=sorted(set(related_articles)),
            chapter=frontmatter.get("chapter") or hierarchy.get("parent"),
            paragraph=frontmatter.get("paragraph") or hierarchy.get("level"),
            legal_domain=frontmatter.get("legal_domain") or frontmatter.get("domain"),
            article_title=article_title,
            graph_rag=graph_rag if isinstance(graph_rag, dict) else {},
            classic_rag=classic_rag if isinstance(classic_rag, dict) else {},
            context_summary=frontmatter.get("summary") or frontmatter.get("context_summary"),
        )


class HybridLegalChunker:
    """
    Hybrid legal document chunking pipeline with metadata/context enrichment.
    """

    def __init__(self):
        self.splitter = SentenceChunker(chunk_size=8, chunk_overlap=1)
        self.refinery = OverlapRefinery()
        self.sectioner = Sectioner()
        self.injector = ContextInjector()
        self.validator = ChunkValidator()
        self.global_chunk_index = 0

    def _make_chunk_id(self, text: str, filepath: str, index: int) -> str:
        """
        Generate deterministic chunk ID.
        """

        raw = f"{filepath}:{index}:{text[:200]}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def _prepare_legal_text(self, text: str) -> str:
        """
        Normalize legal text for better chunking.
        """

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"---+", "", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def process_section(
        self,
        section: dict,
        base_metadata: ChunkMetadata,
        filepath: str,
    ) -> list[Chunk]:
        """
        Convert a single section into RAG chunks.
        """

        header = section["header"]
        raw_text = "\n".join(section["content"]).strip()

        if not raw_text:
            return []

        text = self.injector.inject(
            article_number=base_metadata.article_number,
            header=header,
            article_title=base_metadata.article_title,
            legal_domain=base_metadata.legal_domain,
            topics=base_metadata.topics,
            keywords=base_metadata.keywords,
            text=raw_text,
        )
        text = self._prepare_legal_text(text)

        chunks = self.refinery.refine(self.splitter.chunk(text))
        results = []

        for chunk in chunks:
            part = chunk.text if hasattr(chunk, "text") else str(chunk)
            part = part.strip()

            if not self.validator.is_valid(part):
                continue

            index = self.global_chunk_index
            chunk_id = self._make_chunk_id(part, filepath, index)

            metadata = ChunkMetadata(
                source=base_metadata.source,
                file=base_metadata.file,
                header=header,
                level=base_metadata.level,
                article_number=base_metadata.article_number,
                chunk_index=index,
                topics=base_metadata.topics,
                keywords=base_metadata.keywords,
                related_articles=base_metadata.related_articles,
                chapter=base_metadata.chapter,
                paragraph=base_metadata.paragraph,
                legal_domain=base_metadata.legal_domain,
                article_title=base_metadata.article_title,
                graph_rag=base_metadata.graph_rag,
                classic_rag=base_metadata.classic_rag,
                context_summary=base_metadata.context_summary,
            )

            results.append(Chunk(chunk_id=chunk_id, text=part, metadata=metadata))
            self.global_chunk_index += 1

        return results

    def create_chunks(
        self,
        sections: list[dict],
        frontmatter: dict,
        filepath: str,
    ) -> list[Chunk]:
        """
        Build all chunks from parsed document sections.
        """

        all_chunks = []

        for section in sections:
            metadata = FrontmatterAdapter.build_metadata(
                frontmatter=frontmatter,
                filepath=filepath,
                header=section["header"],
                level=section["level"],
                chunk_index=self.global_chunk_index,
            )

            all_chunks.extend(self.process_section(section, metadata, filepath))

        return all_chunks

    def process(self, filepath: str, frontmatter: dict, body: str) -> list[Chunk]:
        """
        Entry point for the chunking pipeline.
        """

        sections = self.sectioner.extract_sections(body)
        return self.create_chunks(sections, frontmatter, filepath)
