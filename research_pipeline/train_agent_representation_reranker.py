"""Train and visualize a grounded agent reranker from backend exports.

This is the research-side bridge for Danang UrbanAgent AI:

1. Backend exports grounded user/persona/POI pairs to
   D:/POI-urban-danang-BE/data/training/agent_representation_pairs_v1.jsonl.
2. This script trains a lightweight representation reranker in poi_urban.
3. It writes metrics and figures that can be used in reports and PISI slides.

The model here is intentionally classical and transparent. It is the first
research baseline before moving to a PyTorch contrastive encoder.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


DEFAULT_INPUT = Path("D:/POI-urban-danang-BE/data/training/agent_representation_pairs_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/agent_representation")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training export: {path}. Run `npm run export:representation-data` in backend first."
        )
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def is_pair_record(record: dict[str, Any]) -> bool:
    return record.get("record_type") in {"query_positive_poi", "query_negative_poi", "persona_memory_positive"}


def build_text(record: dict[str, Any]) -> str:
    poi = record.get("poi") or {}
    context = record.get("query_context") or {}
    pieces = [
        str(record.get("query") or ""),
        str(record.get("target_role") or ""),
        str(poi.get("name") or ""),
        str(poi.get("category") or ""),
        str(poi.get("district") or ""),
        str(poi.get("semantic_text") or ""),
        json.dumps(context, ensure_ascii=False, sort_keys=True),
    ]
    return " | ".join(piece for piece in pieces if piece)


def numeric_features(records: list[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        poi = record.get("poi") or {}
        numeric = poi.get("numeric_features") or {}
        rating = float(numeric.get("rating") or 0.0)
        review_count = float(numeric.get("review_count") or 0.0)
        rows.append(
            [
                rating / 5.0 if rating <= 5 else rating / 10.0,
                math.log1p(max(review_count, 0.0)) / 10.0,
                1.0 if record.get("record_type") == "persona_memory_positive" else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def make_feature_union(records_for_shape: list[dict[str, Any]]) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "text",
                TfidfVectorizer(
                    preprocessor=build_text,
                    token_pattern=r"(?u)\b\w\w+\b",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=6000,
                ),
            ),
            (
                "numeric",
                FunctionTransformer(lambda rows: numeric_features(list(rows)), validate=False),
            ),
        ]
    )


def ranking_metrics(records: list[dict[str, Any]], scores: np.ndarray) -> dict[str, float]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for record, score in zip(records, scores):
        query_key = str(record.get("sample_id") or record.get("persona_id") or record.get("query"))
        grouped[query_key].append((int(record.get("label") or 0), float(score)))

    recall_at_5: list[float] = []
    precision_at_5: list[float] = []
    mrr: list[float] = []
    for items in grouped.values():
        positives = sum(label for label, _ in items)
        if positives <= 0:
            continue
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        top5 = ranked[:5]
        hits = sum(label for label, _ in top5)
        recall_at_5.append(hits / positives)
        precision_at_5.append(hits / max(len(top5), 1))
        first_hit_rank = next((idx + 1 for idx, (label, _) in enumerate(ranked) if label == 1), None)
        mrr.append(0.0 if first_hit_rank is None else 1.0 / first_hit_rank)

    return {
        "recall_at_5": float(np.mean(recall_at_5)) if recall_at_5 else 0.0,
        "precision_at_5": float(np.mean(precision_at_5)) if precision_at_5 else 0.0,
        "mrr": float(np.mean(mrr)) if mrr else 0.0,
    }


def evaluate(records: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = ranking_metrics(records, scores)
    if len(set(labels.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = 0.0
        metrics["average_precision"] = 0.0
    return metrics


def plot_metrics(metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = ["roc_auc", "average_precision", "recall_at_5", "precision_at_5", "mrr"]
    values = [float(metrics["test"].get(key, 0.0)) for key in keys]

    if plt is None:
        width = 980
        height = 520
        chart_x = 90
        chart_y = 90
        chart_w = 820
        chart_h = 330
        bar_w = 112
        gap = 52
        colors = ["#2563eb", "#0891b2", "#059669", "#d97706", "#7c3aed"]
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<text x="42" y="48" fill="#0f172a" font-family="Arial, sans-serif" font-size="24" font-weight="800">Danang UrbanAgent Representation Reranker</text>',
            '<text x="42" y="74" fill="#475569" font-family="Arial, sans-serif" font-size="14">Test metrics from grounded backend export</text>',
            f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#334155" stroke-width="1"/>',
        ]
        for tick in [0, 0.25, 0.5, 0.75, 1.0]:
            y = chart_y + chart_h - tick * chart_h
            lines.append(f'<line x1="{chart_x}" y1="{y}" x2="{chart_x + chart_w}" y2="{y}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 5"/>')
            lines.append(f'<text x="42" y="{y + 4}" fill="#64748b" font-family="Arial, sans-serif" font-size="12">{tick:.2f}</text>')
        for index, (key, value) in enumerate(zip(keys, values)):
            x = chart_x + index * (bar_w + gap) + 24
            bar_h = max(value, 0.0) * chart_h
            y = chart_y + chart_h - bar_h
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="8" fill="{colors[index]}"/>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{y - 10}" fill="#0f172a" font-family="Arial, sans-serif" font-size="13" font-weight="700" text-anchor="middle">{value:.3f}</text>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{chart_y + chart_h + 30}" fill="#334155" font-family="Arial, sans-serif" font-size="12" text-anchor="middle">{key}</text>')
        lines.append('</svg>')
        (output_dir / "agent_representation_metrics.svg").write_text("\n".join(lines), encoding="utf-8")
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2563eb", "#0891b2", "#059669", "#d97706", "#7c3aed"]
    ax.bar(keys, values, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_title("Danang UrbanAgent Representation Reranker - Test Metrics", fontsize=14, weight="bold")
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    for idx, value in enumerate(values):
        ax.text(idx, min(value + 0.035, 1.02), f"{value:.3f}", ha="center", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "agent_representation_metrics.png", dpi=180)
    plt.close(fig)


def train(input_path: Path, output_dir: Path) -> dict[str, Any]:
    raw_records = read_jsonl(input_path)
    records = [record for record in raw_records if is_pair_record(record) and record.get("label") in {0, 1}]
    if len(records) < 8:
        raise ValueError(f"Need at least 8 labeled pair records, got {len(records)}.")

    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    train_records, test_records, y_train, y_test = train_test_split(
        records,
        labels,
        test_size=0.35,
        random_state=42,
        stratify=labels if len(set(labels.tolist())) > 1 else None,
    )

    feature_union = make_feature_union(train_records)
    x_train = feature_union.fit_transform(train_records)
    x_test = feature_union.transform(test_records)

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)

    train_scores = model.predict_proba(x_train)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]

    metrics = {
        "train": evaluate(train_records, y_train, train_scores),
        "test": evaluate(test_records, y_test, test_scores),
        "data": {
            "input_file": str(input_path),
            "raw_records": len(raw_records),
            "pair_records": len(records),
            "positive_records": int(labels.sum()),
            "negative_records": int((labels == 0).sum()),
            "train_records": len(train_records),
            "test_records": len(test_records),
        },
        "model": {
            "type": "tfidf_numeric_logistic_regression",
            "purpose": "research baseline before PyTorch contrastive fine-tuning",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "agent_representation_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    plot_metrics(metrics, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    metrics = train(args.input, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
