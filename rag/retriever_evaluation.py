import json
from typing import List

import pandas as pd


def recall_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not relevant:
        return 0.0

    return float(
        len(set(retrieved) & set(relevant)) > 0
    )


def precision_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not retrieved:
        return 0.0

    hits = sum(
        1
        for a in retrieved
        if a in relevant
    )

    return hits / len(retrieved)


def mrr_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    for i, a in enumerate(retrieved):

        if a in relevant:
            return 1.0 / (i + 1)

    return 0.0


def evaluate_rag(
    rag,
    dataset,
    output_path="rag_eval_results.json"
):

    results = []

    metrics = {
        "recall": [],
        "precision": [],
        "mrr": []
    }

    print("\n" + "=" * 80)
    print("STARTING DATASET EVALUATION")
    print("=" * 80)

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

        print("\n" + "-" * 80)
        print(f"QUERY: {query}")

        hits = rag.retriever.retrieve(
            query=query,
            top_k=5
        )

        retrieved_articles = []

        contexts = []

        for h in hits:

            article = None

            if h.payload:
                article = h.payload.get(
                    "article_number"
                )

            if article is not None:

                try:
                    retrieved_articles.append(
                        int(article)
                    )
                except Exception:
                    pass

            contexts.append(h.text)

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

        metrics["recall"].append(recall)

        metrics["precision"].append(precision)

        metrics["mrr"].append(mrr)

        print(
            f"RELEVANT: {relevant_articles}"
        )

        print(
            f"RETRIEVED: {retrieved_articles}"
        )

        print(
            f"RECALL@5: {recall:.4f}"
        )

        print(
            f"PRECISION@5: {precision:.4f}"
        )

        print(
            f"MRR@5: {mrr:.4f}"
        )

        results.append({

            "question": query,

            "relevant_articles":
                relevant_articles,

            "hard_negatives":
                hard_negatives,

            "retrieved_articles":
                retrieved_articles,

            "recall@5":
                recall,

            "precision@5":
                precision,

            "mrr@5":
                mrr,

            "contexts":
                contexts
        })

    report = {

        "recall@5":
            sum(metrics["recall"]) / len(dataset),

        "precision@5":
            sum(metrics["precision"]) / len(dataset),

        "mrr@5":
            sum(metrics["mrr"]) / len(dataset),
    }

    print("\n" + "=" * 80)
    print("FINAL RETRIEVER METRICS")
    print("=" * 80)

    print(
        json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        )
    )

    output = {

        "retriever_metrics":
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