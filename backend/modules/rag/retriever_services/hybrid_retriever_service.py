from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HybridRetrieverConfig:
    """
    Configuration for Dense + BM25 + metadata GraphRAG hybrid retrieval.

    The config is intentionally domain-neutral, so the retriever can work
    not only with labor law, but also with civil, criminal, administrative,
    tax or other legal corpora.
    """

    alpha: float = 0.8
    graph_weight: float = 0.15
    pool_multiplier: int = 8
    max_pool_size: int = 80
    min_text_len: int = 40

    document_id_keys: tuple[str, ...] = (
        "article_number",
        "article",
        "article_id",
        "norm_id",
        "document_id",
        "section_number",
        "paragraph_number",
    )

    header_keys: tuple[str, ...] = (
        "header",
        "title",
        "heading",
        "name",
    )

    topic_keys: tuple[str, ...] = (
        "topics",
    )

    nested_topic_keys: tuple[str, ...] = (
        "classic_rag",
    )

    graph_metadata_key: str = "graph_rag"

    relation_target_keys: tuple[str, ...] = (
        "target",
        "to",
        "article",
        "article_number",
        "document_id",
        "norm_id",
    )
