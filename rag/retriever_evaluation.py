from __future__ import annotations

import json
import math
from collections import defaultdict
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
    Precision@k for retrieval.

    For article-level retrieval this must divide by fixed k,
    not by the number of unique retrieved articles.
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
    HitRate@k: at least one relevant article in top-k.
    """

    top_k = retrieved[:k]
    relevant_set = set(relevant)

    return float(
        any(article in relevant_set for article in top_k)
    )


def reciprocal_rank(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Reciprocal rank of first relevant article.
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
        k=k
    )

    if idcg == 0:
        return 0.0

    return dcg_at_k(
        retrieved=retrieved,
        relevant=relevant_unique,
        k=k
    ) / idcg


def hard_negative_rate_at_k(
        retrieved: list[str],
        hard_negatives: list[str],
        k: int
) -> float:
    """
    Share of top-k results that are known hard negatives.
    """

    if k <= 0:
        return 0.0

    hard_negative_set = set(hard_negatives)

    return sum(
        1 for article in retrieved[:k]
        if article in hard_negative_set
    ) / k


def evaluate_ranked_articles(
        retrieved: list[str],
        relevant: list[str],
        hard_negatives: list[str],
        k: int
) -> dict[str, float]:
    """
    Calculate a complete metric bundle for a ranked article list.
    """

    return {
        f"precision@{k}": precision_at_k(retrieved, relevant, k),
        f"recall@{k}": recall_at_k(retrieved, relevant, k),
        f"hitrate@{k}": hitrate_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant, k),
        f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
        f"hard_negative_rate@{k}": hard_negative_rate_at_k(
            retrieved=retrieved,
            hard_negatives=hard_negatives,
            k=k
        ),
    }


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
            4
        )
        for key in keys
    }


def evaluate_rag(
        rag,
        dataset,
        output_path: str = "rag_eval_results.json",
        use_reranker: bool = True,
        retrieve_top_k: int = 20,
        rerank_top_n: int = 5
) -> dict[str, Any]:
    """
    Evaluate RAG retrieval.

    Main metric is article-level retrieval because the dataset is labeled
    by legal articles, not by exact chunk ids.

    Also stores raw chunk sequence for debugging, but raw chunk metrics are
    not treated as the main score because duplicate chunks of one article
    can distort recall and nDCG.
    """

    samples: list[dict[str, Any]] = []
    article_metric_rows: list[dict[str, float]] = []
    chunk_article_metric_rows: list[dict[str, float]] = []
    metrics_by_category: dict[str, list[dict[str, float]]] = defaultdict(list)

    print("\n" + "=" * 100)
    print("STARTING RETRIEVAL EVALUATION")
    print("=" * 100)

    print(f"\nRERANKER ENABLED: {use_reranker}")
    print(f"RETRIEVE TOP-K: {retrieve_top_k}")
    print(f"FINAL TOP-N: {rerank_top_n}")

    for item in dataset:
        query = item["question"]
        category = item.get("category", "unknown")

        relevant_articles = [
            normalize_article_id(article)
            for article in item.get("relevant_articles", [])
        ]

        relevant_articles = [
            article for article in relevant_articles
            if article is not None
        ]

        hard_negatives = [
            normalize_article_id(article)
            for article in item.get("hard_negatives", [])
        ]

        hard_negatives = [
            article for article in hard_negatives
            if article is not None
        ]

        hits = rag.search(
            query=query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_n=rerank_top_n,
            use_reranker=use_reranker
        )

        retrieved_articles_raw: list[str] = []
        retrieved_chunks: list[dict[str, Any]] = []

        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}

            article = normalize_article_id(
                payload.get("article_number")
            )

            if article is not None:
                retrieved_articles_raw.append(article)

            retrieved_chunks.append({
                "rank": rank,
                "article": article,
                "score": round(float(hit.score or 0.0), 4),
                "sources": payload.get("retrieval_sources")
                           or payload.get("retrieval_source"),
                "header": payload.get("header"),
                "text": (hit.text or "")[:1500]
            })

        # Main metric: article-level unique ranking.
        retrieved_articles = unique_preserve_order(
            retrieved_articles_raw
        )[:rerank_top_n]

        # Debug metric: chunk order but with binary article matching.
        # This shows how duplicates affect the actual shown top-k.
        retrieved_chunk_articles = retrieved_articles_raw[:rerank_top_n]

        article_metrics = evaluate_ranked_articles(
            retrieved=retrieved_articles,
            relevant=relevant_articles,
            hard_negatives=hard_negatives,
            k=rerank_top_n
        )

        chunk_article_metrics = evaluate_ranked_articles(
            retrieved=retrieved_chunk_articles,
            relevant=relevant_articles,
            hard_negatives=hard_negatives,
            k=rerank_top_n
        )

        article_metric_rows.append(article_metrics)
        chunk_article_metric_rows.append(chunk_article_metrics)
        metrics_by_category[category].append(article_metrics)

        print("\n" + "-" * 100)
        print(f"QUERY: {query}")
        print(f"RELEVANT: {relevant_articles}")
        print(f"HARD NEGATIVES: {hard_negatives}")
        print(f"RETRIEVED ARTICLES: {retrieved_articles}")
        print(
            json.dumps(
                article_metrics,
                indent=4,
                ensure_ascii=False
            )
        )

        samples.append({
            "id": item.get("id"),
            "question": query,
            "category": category,
            "relevant_articles": relevant_articles,
            "hard_negatives": hard_negatives,
            "retrieved_articles": retrieved_articles,
            "retrieved_articles_chunk_order": retrieved_chunk_articles,
            "article_level_metrics": article_metrics,
            "chunk_order_article_metrics": chunk_article_metrics,
            "retrieved_chunks": retrieved_chunks
        })

    article_level_report = _mean_metric(article_metric_rows)
    chunk_order_article_report = _mean_metric(chunk_article_metric_rows)

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
                "article_level_metrics are primary because the dataset "
                "is labeled by legal article ids."
            )
        },
        "retrieval_metrics": article_level_report,
        "article_level_metrics": article_level_report,
        "chunk_order_article_metrics": chunk_order_article_report,
        "category_metrics": category_report,
        "samples": samples
    }

    print("\n" + "=" * 100)
    print("FINAL ARTICLE-LEVEL RETRIEVAL METRICS")
    print("=" * 100)
    print(json.dumps(article_level_report, indent=4, ensure_ascii=False))

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    pd.DataFrame(samples).to_csv(
        output_path.replace(".json", ".csv"),
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved -> {output_path}")

    return output
