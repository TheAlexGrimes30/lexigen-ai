import json
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


def recall_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not relevant:
        return 0.0

    hits = set(retrieved) & set(relevant)

    return float(len(hits) > 0)


def precision_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    if not retrieved:
        return 0.0

    hits = sum(
        1
        for article in retrieved
        if article in relevant
    )

    return hits / len(retrieved)


def mrr_at_k(
    retrieved: List[int],
    relevant: List[int]
) -> float:

    for i, article in enumerate(retrieved):

        if article in relevant:
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
            top_k=10
        )

        retrieved_articles = []

        retrieved_chunks = []

        for rank, h in enumerate(
            hits,
            start=1
        ):

            article = None

            if h.payload:

                article = h.payload.get(
                    "article_number"
                )

            try:

                if article is not None:

                    article = int(article)

                    retrieved_articles.append(
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

        # IMPORTANT:
        # remove duplicate articles
        # while preserving ranking order

        unique_articles = (
            unique_preserve_order(
                retrieved_articles
            )
        )

        # METRICS

        recall = recall_at_k(
            unique_articles,
            relevant_articles
        )

        precision = precision_at_k(
            unique_articles,
            relevant_articles
        )

        mrr = mrr_at_k(
            unique_articles,
            relevant_articles
        )

        metrics["recall"].append(
            recall
        )

        metrics["precision"].append(
            precision
        )

        metrics["mrr"].append(
            mrr
        )

        print(
            f"RELEVANT: "
            f"{relevant_articles}"
        )

        print(
            f"RETRIEVED RAW: "
            f"{retrieved_articles}"
        )

        print(
            f"RETRIEVED UNIQUE: "
            f"{unique_articles}"
        )

        print(
            f"RECALL@5: "
            f"{recall:.4f}"
        )

        print(
            f"PRECISION@5: "
            f"{precision:.4f}"
        )

        print(
            f"MRR@5: "
            f"{mrr:.4f}"
        )

        results.append({

            "question":
                query,

            "relevant_articles":
                relevant_articles,

            "hard_negatives":
                hard_negatives,

            "retrieved_articles_raw":
                retrieved_articles,

            "retrieved_articles_unique":
                unique_articles,

            "recall@5":
                recall,

            "precision@5":
                precision,

            "mrr@5":
                mrr,

            "retrieved_chunks":
                retrieved_chunks
        })

    report = {

        "recall@5":
            round(
                sum(metrics["recall"])
                / len(dataset),
                4
            ),

        "precision@5":
            round(
                sum(metrics["precision"])
                / len(dataset),
                4
            ),

        "mrr@5":
            round(
                sum(metrics["mrr"])
                / len(dataset),
                4
            )
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

    # CSV export

    csv_rows = []

    for sample in results:

        csv_rows.append({

            "question":
                sample["question"],

            "relevant_articles":
                sample[
                    "relevant_articles"
                ],

            "retrieved_articles_unique":
                sample[
                    "retrieved_articles_unique"
                ],

            "recall@5":
                sample["recall@5"],

            "precision@5":
                sample["precision@5"],

            "mrr@5":
                sample["mrr@5"]
        })

    pd.DataFrame(csv_rows).to_csv(

        output_path.replace(
            ".json",
            ".csv"
        ),

        index=False,
        encoding="utf-8"
    )

    print(
        f"\nSaved -> {output_path}"
    )