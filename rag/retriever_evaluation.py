from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def normalize_article_id(value: Any) -> str | None:
    """
    Normalize legal article id for evaluation.

    Important:
    - keeps values as strings;
    - preserves decimal article ids like "307.1";
    - removes only useless ".0" suffix.
    """

    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    if text.endswith(".0"):
        text = text[:-2]

    return text


def normalize_article_list(values: Iterable[Any]) -> list[str]:
    """
    Normalize and deduplicate article ids while preserving order.
    """

    return unique_preserve_order(
        normalize_article_id(value)
        for value in values
    )


def unique_preserve_order(items: Iterable[str | None]) -> list[str]:
    """
    Remove duplicates while preserving the original ranking order.
    """

    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if item is None:
            continue

        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@k for article-level retrieval.

    The denominator is fixed k. If fewer than k unique articles are available
    because the top chunks repeat the same article, precision is intentionally
    penalized. This makes duplicate-heavy retrieval visible in the metric.
    """

    if k <= 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = sum(
        1 for article in top_k
        if article in relevant_set
    )

    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Recall@k with binary article matching.

    Duplicate chunks of the same relevant article must not produce recall > 1.
    """

    if not relevant:
        return 0.0

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    hits = {
        article for article in top_k
        if article in relevant_set
    }

    return len(hits) / len(relevant_set)


def hitrate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    HitRate@k: whether at least one relevant article appears in top-k.
    """

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    return float(
        any(article in relevant_set for article in top_k)
    )


def reciprocal_rank(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Reciprocal rank of the first relevant article in top-k.
    """

    relevant_set = set(relevant)

    for rank, article in enumerate(retrieved[:k], start=1):
        if article in relevant_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    DCG@k with duplicate relevant articles counted only once.
    """

    relevant_set = set(relevant)
    seen_relevant: set[str] = set()
    score = 0.0

    for rank, article in enumerate(retrieved[:k], start=1):
        if article not in relevant_set:
            continue

        if article in seen_relevant:
            continue

        score += 1.0 / math.log2(rank + 1)
        seen_relevant.add(article)

    return score


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    nDCG@k in range 0..1.
    """

    if not relevant:
        return 0.0

    relevant_unique = unique_preserve_order(relevant)
    ideal = relevant_unique[:k]

    idcg = dcg_at_k(
        retrieved=ideal,
        relevant=relevant_unique,
        k=k,
    )

    if idcg == 0:
        return 0.0

    return dcg_at_k(
        retrieved=retrieved,
        relevant=relevant_unique,
        k=k,
    ) / idcg


def hard_negative_rate_at_k(
    retrieved: list[str],
    hard_negatives: list[str],
    k: int,
) -> float:
    """
    Share of top-k article results that are known hard negatives.
    """

    if k <= 0:
        return 0.0

    hard_negative_set = set(hard_negatives)

    return sum(
        1 for article in retrieved[:k]
        if article in hard_negative_set
    ) / k


def duplicate_article_rate_at_k(retrieved_chunk_articles: list[str], k: int) -> float:
    """
    Share of top-k chunk results wasted by repeated article ids.

    Example:
    ["307", "307", "307", "308", "322"] -> duplicate rate is 2/5.
    """

    if k <= 0:
        return 0.0

    top_k = retrieved_chunk_articles[:k]

    if not top_k:
        return 0.0

    unique_count = len(unique_preserve_order(top_k))
    duplicate_count = len(top_k) - unique_count

    return duplicate_count / k


def evaluate_ranked_articles(
    retrieved: list[str],
    relevant: list[str],
    hard_negatives: list[str],
    k: int,
) -> dict[str, float]:
    """
    Calculate article-level retrieval metrics for a ranked article list.
    """

    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"hitrate@{k}": hitrate_at_k(retrieved, relevant, k),
        f"mrr@{k}": reciprocal_rank(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
        f"hard_negative_rate@{k}": hard_negative_rate_at_k(
            retrieved=retrieved,
            hard_negatives=hard_negatives,
            k=k,
        ),
    }


def evaluate_chunk_article_diagnostics(
    retrieved_chunk_articles: list[str],
    relevant: list[str],
    hard_negatives: list[str],
    k: int,
) -> dict[str, float]:
    """
    Calculate diagnostics on the raw chunk order.

    These values are not the main retrieval score. They help detect cases where
    several chunks from one article occupy the whole top-k.
    """

    metrics = evaluate_ranked_articles(
        retrieved=retrieved_chunk_articles,
        relevant=relevant,
        hard_negatives=hard_negatives,
        k=k,
    )

    metrics[f"duplicate_article_rate@{k}"] = duplicate_article_rate_at_k(
        retrieved_chunk_articles=retrieved_chunk_articles,
        k=k,
    )

    return metrics


def _mean_metric(rows: list[dict[str, float]]) -> dict[str, float]:
    """
    Average metric dictionaries.
    """

    if not rows:
        return {}

    keys = rows[0].keys()

    return {
        key: round(
            sum(row[key] for row in rows) / len(rows),
            4,
        )
        for key in keys
    }


def _article_from_hit(hit: Any) -> str | None:
    """
    Extract article id from a SearchResult-like hit.
    """

    payload = getattr(hit, "payload", None) or {}

    return normalize_article_id(
        payload.get("retrieval_doc_id")
        or payload.get("article_number")
        or payload.get("article")
        or payload.get("article_id")
        or payload.get("norm_id")
    )


def _serialize_hit(rank: int, hit: Any, article: str | None) -> dict[str, Any]:
    """
    Serialize a retrieved chunk for debugging output.
    """

    payload = getattr(hit, "payload", None) or {}
    text = getattr(hit, "text", "") or ""
    score = getattr(hit, "score", 0.0) or 0.0

    return {
        "rank": rank,
        "article": article,
        "score": round(float(score), 4),
        "sources": payload.get("retrieval_sources")
        or payload.get("retrieval_source"),
        "header": payload.get("header"),
        "chunk_index": payload.get("chunk_index"),
        "text": text[:1500],
    }


def _search_rag(
    rag: Any,
    *,
    query: str,
    retrieve_top_k: int,
    rerank_top_n: int,
    use_reranker: bool,
) -> list[Any]:
    """
    Execute project RAG search.
    """

    return rag.search(
        query=query,
        retrieve_top_k=retrieve_top_k,
        rerank_top_n=rerank_top_n,
        use_reranker=use_reranker,
    )


def evaluate_rag(
    rag: Any,
    dataset: list[dict[str, Any]],
    output_path: str = "rag_eval_results.json",
    use_reranker: bool = True,
    retrieve_top_k: int = 20,
    rerank_top_n: int = 5,
) -> dict[str, Any]:
    """
    Evaluate RAG retrieval on article-level labels.

    Main score:
    - converts retrieved chunks to article ids;
    - removes duplicate article ids while preserving ranking order;
    - computes Precision@K, Recall@K, HitRate@K, MRR@K and nDCG@K.

    Diagnostics:
    - preserves raw chunk article order;
    - reports duplicate_article_rate@K to show whether one article occupies
      several top-k chunk positions.
    """

    samples: list[dict[str, Any]] = []
    article_metric_rows: list[dict[str, float]] = []
    chunk_diagnostic_rows: list[dict[str, float]] = []
    metrics_by_category: dict[str, list[dict[str, float]]] = defaultdict(list)

    print("\n" + "=" * 100)
    print("STARTING ARTICLE-LEVEL RETRIEVAL EVALUATION")
    print("=" * 100)

    print(f"\nRERANKER ENABLED: {use_reranker}")
    print(f"RETRIEVE TOP-K: {retrieve_top_k}")
    print(f"FINAL TOP-N / METRIC K: {rerank_top_n}")

    for item in dataset:
        query = item["question"]
        category = item.get("category", "unknown")

        relevant_articles = normalize_article_list(
            item.get("relevant_articles", [])
        )
        hard_negatives = normalize_article_list(
            item.get("hard_negatives", [])
        )

        hits = _search_rag(
            rag,
            query=query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_n=rerank_top_n,
            use_reranker=use_reranker,
        )

        retrieved_chunk_articles: list[str] = []
        retrieved_chunks: list[dict[str, Any]] = []

        for rank, hit in enumerate(hits, start=1):
            article = _article_from_hit(hit)

            if article is not None:
                retrieved_chunk_articles.append(article)

            retrieved_chunks.append(
                _serialize_hit(
                    rank=rank,
                    hit=hit,
                    article=article,
                )
            )

        # Main metric: unique article ranking, not chunk ranking.
        retrieved_articles = unique_preserve_order(
            retrieved_chunk_articles
        )

        article_metrics = evaluate_ranked_articles(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            hard_negatives=hard_negatives,
            k=rerank_top_n,
        )

        chunk_diagnostics = evaluate_chunk_article_diagnostics(
            retrieved_chunk_articles=retrieved_chunk_articles,
            relevant=relevant_articles,
            hard_negatives=hard_negatives,
            k=rerank_top_n,
        )

        article_metric_rows.append(article_metrics)
        chunk_diagnostic_rows.append(chunk_diagnostics)
        metrics_by_category[category].append(article_metrics)

        print("\n" + "-" * 100)
        print(f"QUERY: {query}")
        print(f"RELEVANT: {relevant_articles}")
        print(f"HARD NEGATIVES: {hard_negatives}")
        print(f"RETRIEVED CHUNK ARTICLES: {retrieved_chunk_articles}")
        print(f"RETRIEVED UNIQUE ARTICLES: {retrieved_articles}")
        print(json.dumps(article_metrics, indent=4, ensure_ascii=False))

        samples.append(
            {
                "id": item.get("id"),
                "question": query,
                "category": category,
                "relevant_articles": relevant_articles,
                "hard_negatives": hard_negatives,
                "retrieved_articles": retrieved_articles,
                "retrieved_chunk_articles": retrieved_chunk_articles,
                "article_level_metrics": article_metrics,
                "chunk_order_diagnostics": chunk_diagnostics,
                "retrieved_chunks": retrieved_chunks,
            }
        )

    article_level_report = _mean_metric(article_metric_rows)
    chunk_diagnostic_report = _mean_metric(chunk_diagnostic_rows)

    category_report = {
        category: _mean_metric(rows)
        for category, rows in metrics_by_category.items()
    }

    output = {
        "config": {
            "use_reranker": use_reranker,
            "retrieve_top_k": retrieve_top_k,
            "rerank_top_n": rerank_top_n,
            "metric_policy": (
                "Primary metrics are article-level: retrieved chunks are "
                "collapsed to unique article ids before scoring because "
                "the dataset is labeled by legal article ids."
            ),
        },
        "retrieval_metrics": article_level_report,
        "article_level_metrics": article_level_report,
        "chunk_order_diagnostics": chunk_diagnostic_report,
        "category_metrics": category_report,
        "samples": samples,
    }

    print("\n" + "=" * 100)
    print("FINAL ARTICLE-LEVEL RETRIEVAL METRICS")
    print("=" * 100)
    print(json.dumps(article_level_report, indent=4, ensure_ascii=False))
    print("\nCHUNK ORDER DIAGNOSTICS")
    print(json.dumps(chunk_diagnostic_report, indent=4, ensure_ascii=False))

    output_file = Path(output_path)

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4,
        )

    pd.DataFrame(samples).to_csv(
        output_file.with_suffix(".csv"),
        index=False,
        encoding="utf-8",
    )

    print(f"\nSaved -> {output_file}")

    return output
