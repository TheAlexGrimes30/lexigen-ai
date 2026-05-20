import json
import math
from typing import List

import pandas as pd


def unique_preserve_order(
    items: List[int]
) -> List[int]:

    seen = set()

    result = []

    for item in items:

        if item not in seen:

            seen.add(item)

            result.append(item)

    return result


def hit_rate_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not relevant:
        return 0.0

    return float(
        len(set(retrieved[:len(retrieved)]) & set(relevant)) > 0
    )


def recall_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not relevant:
        return 0.0

    hits = len(
        set(retrieved) & set(relevant)
    )

    return hits / len(relevant)


def precision_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not retrieved:
        return 0.0

    retrieved_unique = set(retrieved)

    hits = len(
        retrieved_unique & set(relevant)
    )

    return hits / len(retrieved_unique)


def mrr_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    for rank, article in enumerate(retrieved, start=1):

        if article in relevant:

            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    dcg = 0.0

    for rank, article in enumerate(retrieved, start=1):

        if article in relevant:

            dcg += 1.0 / math.log2(rank + 1)

    return dcg


def ideal_dcg_at_k(
    num_relevant: int,
    k: int
) -> float:

    ideal_hits = min(num_relevant, k)

    return sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )


def ndcg_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not relevant:
        return 0.0

    dcg = dcg_at_k(
        retrieved,
        relevant
    )

    idcg = ideal_dcg_at_k(
        num_relevant=len(relevant),
        k=len(retrieved)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg


def evaluate_rag(
    rag,
    dataset,
    output_path="rag_eval_results.json",

    use_reranker=True,

    retrieve_top_k=20,
    rerank_top_n=5
):

    results = []

    metrics = {

        "hit_rate": [],
        "recall": [],
        "precision": [],
        "mrr": [],
        "ndcg": []
    }

    print("\n" + "=" * 100)

    print("STARTING DATASET EVALUATION")

    print("=" * 100)

    print(f"\nRERANKER ENABLED: {use_reranker}")
    print(f"RETRIEVE TOP-K: {retrieve_top_k}")
    print(f"FINAL TOP-N: {rerank_top_n}")

    for item in dataset:

        query = item["question"]

        relevant_articles = item.get(
            "relevant_articles",
            []
        )

        hard_negatives = item.get(
            "hard_negatives",
            []
        )

        print("\n" + "-" * 100)

        print(f"QUERY: {query}")

        hits = rag.search(
            query=query,
            retrieve_top_k=retrieve_top_k,
            rerank_top_n=rerank_top_n,
            use_reranker=use_reranker
        )

        retrieved_articles_raw = []

        retrieved_chunks = []

        for rank, h in enumerate(hits, start=1):

            article = None

            if h.payload:

                article = h.payload.get(
                    "article_number"
                )

            try:

                if article is not None:

                    article = int(article)

                    retrieved_articles_raw.append(
                        article
                    )

            except Exception:
                continue

            retrieved_chunks.append({

                "rank":
                    rank,

                "article":
                    article,

                "score":
                    round(h.score, 4),

                "text":
                    (h.text or "")[:1500]
            })


        retrieved_articles = (
            unique_preserve_order(
                retrieved_articles_raw
            )
        )


        retrieved_articles = (
            retrieved_articles[:rerank_top_n]
        )


        hit_rate = hit_rate_at_k(
            retrieved_articles,
            relevant_articles
        )

        recall = recall_at_k(
            retrieved_articles,
            relevant_articles
        )

        precision = precision_at_k(
            retrieved_articles,
            relevant_articles
        )

        mrr = mrr_at_k(
            retrieved_articles,
            relevant_articles
        )

        ndcg = ndcg_at_k(
            retrieved_articles,
            relevant_articles
        )

        metrics["hit_rate"].append(hit_rate)
        metrics["recall"].append(recall)
        metrics["precision"].append(precision)
        metrics["mrr"].append(mrr)
        metrics["ndcg"].append(ndcg)


        print(f"RELEVANT: {relevant_articles}")

        print(
            f"RETRIEVED RAW: "
            f"{retrieved_articles_raw}"
        )

        print(
            f"RETRIEVED UNIQUE: "
            f"{retrieved_articles}"
        )

        print(
            f"HITRATE@{rerank_top_n}: "
            f"{hit_rate:.4f}"
        )

        print(
            f"RECALL@{rerank_top_n}: "
            f"{recall:.4f}"
        )

        print(
            f"PRECISION@{rerank_top_n}: "
            f"{precision:.4f}"
        )

        print(
            f"MRR@{rerank_top_n}: "
            f"{mrr:.4f}"
        )

        print(
            f"nDCG@{rerank_top_n}: "
            f"{ndcg:.4f}"
        )


        results.append({

            "question":
                query,

            "relevant_articles":
                relevant_articles,

            "hard_negatives":
                hard_negatives,

            "retrieved_articles_raw":
                retrieved_articles_raw,

            "retrieved_articles_unique":
                retrieved_articles,

            f"hit_rate@{rerank_top_n}":
                hit_rate,

            f"recall@{rerank_top_n}":
                recall,

            f"precision@{rerank_top_n}":
                precision,

            f"mrr@{rerank_top_n}":
                mrr,

            f"ndcg@{rerank_top_n}":
                ndcg,

            "retrieved_chunks":
                retrieved_chunks
        })


    report = {

        f"hit_rate@{rerank_top_n}":
            round(sum(metrics["hit_rate"]) / len(dataset), 4),

        f"recall@{rerank_top_n}":
            round(sum(metrics["recall"]) / len(dataset), 4),

        f"precision@{rerank_top_n}":
            round(sum(metrics["precision"]) / len(dataset), 4),

        f"mrr@{rerank_top_n}":
            round(sum(metrics["mrr"]) / len(dataset), 4),

        f"ndcg@{rerank_top_n}":
            round(sum(metrics["ndcg"]) / len(dataset), 4)
    }

    print("\n" + "=" * 100)

    print("FINAL RETRIEVAL METRICS")

    print("=" * 100)

    print(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        )
    )

    output = {

        "config": {

            "use_reranker":
                use_reranker,

            "retrieve_top_k":
                retrieve_top_k,

            "rerank_top_n":
                rerank_top_n
        },

        "retrieval_metrics":
            report,

        "samples":
            results
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=4
        )

    pd.DataFrame(results).to_csv(

        output_path.replace(
            ".json",
            ".csv"
        ),

        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved -> {output_path}")