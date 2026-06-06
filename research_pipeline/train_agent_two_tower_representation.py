"""Train a small two-tower representation model for Danang UrbanAgent AI.

This is the first "real" neural representation baseline for the agent:
- Query tower encodes user intent/persona context.
- POI tower encodes grounded POI semantic text + numeric signals.
- Dot product predicts whether the POI fits the query/persona.

It is intentionally lightweight so it can run on CPU for PISI demos. Later, the
same training data can be reused with CLIP/SentenceTransformer encoders and
SemanticAwareContrastiveLoss.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


DEFAULT_INPUT = Path("D:/POI-urban-danang-BE/data/training/agent_representation_pairs_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("results/agent_representation_two_tower")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing training export: {path}. Run `npm run export:representation-data` in backend first."
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_pair_record(record: dict[str, Any]) -> bool:
    return record.get("record_type") in {"query_positive_poi", "query_negative_poi", "persona_memory_positive"}


def query_text(record: dict[str, Any]) -> str:
    context = record.get("query_context") or {}
    return " | ".join(
        [
            str(record.get("query") or ""),
            str(record.get("target_role") or ""),
            json.dumps(context, ensure_ascii=False, sort_keys=True),
        ]
    )


def poi_text(record: dict[str, Any]) -> str:
    poi = record.get("poi") or {}
    return " | ".join(
        [
            str(poi.get("name") or ""),
            str(poi.get("category") or ""),
            str(poi.get("district") or ""),
            str(poi.get("semantic_text") or ""),
        ]
    )


def poi_numeric(record: dict[str, Any]) -> list[float]:
    poi = record.get("poi") or {}
    numeric = poi.get("numeric_features") or {}
    rating = float(numeric.get("rating") or 0.0)
    review_count = float(numeric.get("review_count") or 0.0)
    return [
        rating / 5.0 if rating <= 5 else rating / 10.0,
        np.log1p(max(review_count, 0.0)) / 10.0,
        1.0 if record.get("record_type") == "persona_memory_positive" else 0.0,
    ]


class TwoTowerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        labels: np.ndarray,
        vectorizer: TfidfVectorizer,
    ) -> None:
        self.records = records
        self.labels = labels.astype(np.float32)
        self.query_matrix = vectorizer.transform([query_text(record) for record in records]).astype(np.float32)
        self.poi_matrix = vectorizer.transform([poi_text(record) for record in records]).astype(np.float32)
        self.numeric = np.asarray([poi_numeric(record) for record in records], dtype=np.float32)
        roles = sorted({str(record.get("target_role") or "unknown") for record in records if int(record.get("label") or 0) == 1})
        self.role_to_label = {role: index for index, role in enumerate(roles)}
        self.supcon_labels = np.asarray(
            [
                self.role_to_label.get(str(record.get("target_role") or "unknown"), -1)
                if int(record.get("label") or 0) == 1
                else -1
                for record in records
            ],
            dtype=np.int64,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        query = torch.from_numpy(self.query_matrix[index].toarray().squeeze(0))
        poi = torch.from_numpy(self.poi_matrix[index].toarray().squeeze(0))
        numeric = torch.from_numpy(self.numeric[index])
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        supcon_label = torch.tensor(self.supcon_labels[index], dtype=torch.long)
        return query, poi, numeric, label, supcon_label


class TwoTowerRepresentation(nn.Module):
    def __init__(self, input_dim: int, numeric_dim: int = 3, embedding_dim: int = 96) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, embedding_dim),
        )
        self.poi_encoder = nn.Sequential(
            nn.Linear(input_dim + numeric_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, embedding_dim),
        )
        self.temperature = nn.Parameter(torch.tensor(0.08))

    def encode(self, query: torch.Tensor, poi: torch.Tensor, numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_emb = F.normalize(self.query_encoder(query), p=2, dim=1)
        poi_input = torch.cat([poi, numeric], dim=1)
        poi_emb = F.normalize(self.poi_encoder(poi_input), p=2, dim=1)
        return query_emb, poi_emb

    def forward(self, query: torch.Tensor, poi: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        query_emb, poi_emb = self.encode(query, poi, numeric)
        temperature = self.temperature.clamp(0.03, 0.5)
        return (query_emb * poi_emb).sum(dim=1) / temperature


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.12) -> torch.Tensor:
    valid = labels >= 0
    features = features[valid]
    labels = labels[valid]
    if features.shape[0] < 2:
        return torch.zeros((), device=features.device)

    features = F.normalize(features, p=2, dim=1)
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)) & ~self_mask
    if not pos_mask.any():
        return torch.zeros((), device=features.device)

    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    pos_count = pos_mask.float().sum(dim=1).clamp_min(1.0)
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_count
    valid_anchor = pos_mask.any(dim=1)
    return -mean_log_prob_pos[valid_anchor].mean()


def predict_scores(model: nn.Module, dataset: TwoTowerDataset) -> np.ndarray:
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    scores: list[float] = []
    with torch.no_grad():
        for query, poi, numeric, _, _ in loader:
            logits = model(query, poi, numeric)
            scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float32)


def ranking_metrics(records: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    groups: dict[str, list[tuple[int, float]]] = {}
    for record, label, score in zip(records, labels, scores):
        key = str(record.get("sample_id") or record.get("persona_id") or record.get("query"))
        groups.setdefault(key, []).append((int(label), float(score)))

    recall_at_5: list[float] = []
    precision_at_5: list[float] = []
    mrr: list[float] = []
    for items in groups.values():
        positives = sum(label for label, _ in items)
        if positives <= 0:
            continue
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        top5 = ranked[:5]
        hits = sum(label for label, _ in top5)
        recall_at_5.append(hits / positives)
        precision_at_5.append(hits / max(len(top5), 1))
        first_hit = next((index + 1 for index, (label, _) in enumerate(ranked) if label == 1), None)
        mrr.append(0.0 if first_hit is None else 1.0 / first_hit)
    return {
        "recall_at_5": float(np.mean(recall_at_5)) if recall_at_5 else 0.0,
        "precision_at_5": float(np.mean(precision_at_5)) if precision_at_5 else 0.0,
        "mrr": float(np.mean(mrr)) if mrr else 0.0,
    }


def evaluate(records: list[dict[str, Any]], labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    metrics = ranking_metrics(records, labels, scores)
    if len(set(labels.tolist())) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, scores))
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    else:
        metrics["roc_auc"] = 0.0
        metrics["average_precision"] = 0.0
    return metrics


def group_key(record: dict[str, Any]) -> str:
    return str(record.get("sample_id") or record.get("persona_id") or record.get("query") or "unknown")


def grouped_train_test_split(
    records: list[dict[str, Any]],
    labels: np.ndarray,
    test_size: float = 0.35,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(group_key(record), []).append(index)

    rng = np.random.default_rng(seed)
    group_names = np.asarray(sorted(groups))
    rng.shuffle(group_names)

    target_test_records = max(1, int(len(records) * test_size))
    test_indices: list[int] = []
    train_indices: list[int] = []
    for name in group_names:
        bucket = groups[str(name)]
        if len(test_indices) < target_test_records:
            test_indices.extend(bucket)
        else:
            train_indices.extend(bucket)

    if not train_indices or not test_indices:
        raise ValueError("Grouped split failed; need more synthetic groups.")

    return (
        [records[index] for index in train_indices],
        [records[index] for index in test_indices],
        labels[train_indices],
        labels[test_indices],
    )


def plot_training(history: list[float], metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if plt is None:
        width = 1120
        height = 560
        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f8fafc"/>',
            '<text x="44" y="48" fill="#0f172a" font-family="Arial, sans-serif" font-size="24" font-weight="800">Danang UrbanAgent Neural Representation Baseline</text>',
            '<text x="44" y="74" fill="#475569" font-family="Arial, sans-serif" font-size="14">Two-tower query/POI embedding training report</text>',
        ]

        loss_x = 72
        loss_y = 112
        loss_w = 460
        loss_h = 330
        max_loss = max(history) if history else 1.0
        min_loss = min(history) if history else 0.0
        denom = max(max_loss - min_loss, 1e-9)
        points = []
        for index, value in enumerate(history):
            x = loss_x + (index / max(len(history) - 1, 1)) * loss_w
            y = loss_y + loss_h - ((value - min_loss) / denom) * loss_h
            points.append(f"{x:.2f},{y:.2f}")
        lines.extend(
            [
                f'<rect x="{loss_x}" y="{loss_y}" width="{loss_w}" height="{loss_h}" fill="#ffffff" stroke="#cbd5e1" rx="12"/>',
                f'<text x="{loss_x}" y="{loss_y - 18}" fill="#0f172a" font-family="Arial, sans-serif" font-size="16" font-weight="700">Training Loss</text>',
                f'<polyline points="{" ".join(points)}" fill="none" stroke="#ef4444" stroke-width="3"/>',
                f'<text x="{loss_x}" y="{loss_y + loss_h + 28}" fill="#64748b" font-family="Arial, sans-serif" font-size="12">Epochs: {len(history)} | final loss: {(history[-1] if history else 0):.6f}</text>',
            ]
        )

        keys = ["roc_auc", "average_precision", "recall_at_5", "precision_at_5", "mrr"]
        values = [float(metrics["test"].get(key, 0.0)) for key in keys]
        chart_x = 620
        chart_y = 112
        chart_w = 420
        chart_h = 330
        bar_w = 58
        gap = 26
        colors = ["#2563eb", "#0891b2", "#059669", "#d97706", "#7c3aed"]
        lines.append(f'<rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" fill="#ffffff" stroke="#cbd5e1" rx="12"/>')
        lines.append(f'<text x="{chart_x}" y="{chart_y - 18}" fill="#0f172a" font-family="Arial, sans-serif" font-size="16" font-weight="700">Test Metrics</text>')
        for index, (key, value) in enumerate(zip(keys, values)):
            x = chart_x + 34 + index * (bar_w + gap)
            bar_h = max(value, 0.0) * (chart_h - 54)
            y = chart_y + chart_h - 34 - bar_h
            lines.append(f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="7" fill="{colors[index]}"/>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{y - 8}" fill="#0f172a" font-family="Arial, sans-serif" font-size="12" font-weight="700" text-anchor="middle">{value:.3f}</text>')
            lines.append(f'<text x="{x + bar_w / 2}" y="{chart_y + chart_h - 12}" fill="#334155" font-family="Arial, sans-serif" font-size="10" text-anchor="middle">{key}</text>')
        lines.append('</svg>')
        (output_dir / "two_tower_training_report.svg").write_text("\n".join(lines), encoding="utf-8")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(range(1, len(history) + 1), history, color="#ef4444", marker="o")
    axes[0].set_title("Two-Tower Training Loss", weight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].grid(True, linestyle="--", alpha=0.25)

    keys = ["roc_auc", "average_precision", "recall_at_5", "precision_at_5", "mrr"]
    values = [float(metrics["test"].get(key, 0.0)) for key in keys]
    axes[1].bar(keys, values, color=["#2563eb", "#0891b2", "#059669", "#d97706", "#7c3aed"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Test Metrics", weight="bold")
    axes[1].tick_params(axis="x", rotation=25)
    for idx, value in enumerate(values):
        axes[1].text(idx, min(value + 0.03, 1.02), f"{value:.3f}", ha="center", fontsize=9, weight="bold")

    fig.suptitle("Danang UrbanAgent Neural Representation Baseline", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "two_tower_training_report.png", dpi=180)
    plt.close(fig)


def train(input_path: Path, output_dir: Path, epochs: int = 80) -> dict[str, Any]:
    set_seed()
    print(f"[1/8] Reading grounded training pairs from {input_path}")
    raw_records = read_jsonl(input_path)
    records = [record for record in raw_records if is_pair_record(record) and record.get("label") in {0, 1}]
    if len(records) < 8:
        raise ValueError(f"Need at least 8 labeled pair records, got {len(records)}.")
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)

    print(f"[2/8] Building grouped train/test split from {len(records)} pair records")
    train_records, test_records, y_train, y_test = grouped_train_test_split(records, labels)

    print("[3/8] Fitting shared TF-IDF vocabulary")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=1,
        max_features=3500,
    )
    vectorizer.fit([query_text(record) for record in train_records] + [poi_text(record) for record in train_records])

    print(f"[4/8] Building torch datasets: train={len(train_records)} test={len(test_records)}")
    train_dataset = TwoTowerDataset(train_records, y_train, vectorizer)
    test_dataset = TwoTowerDataset(test_records, y_test, vectorizer)
    loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = TwoTowerRepresentation(input_dim=len(vectorizer.vocabulary_))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    pos_weight_value = max(float((y_train == 0).sum()) / max(float(y_train.sum()), 1.0), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value))

    history: list[float] = []
    model.train()
    print(f"[5/8] Training two-tower representation for {epochs} epochs")
    for epoch in range(epochs):
        losses: list[float] = []
        for query, poi, numeric, label, supcon_label in loader:
            optimizer.zero_grad()
            query_emb, poi_emb = model.encode(query, poi, numeric)
            temperature = model.temperature.clamp(0.03, 0.5)
            logits = (query_emb * poi_emb).sum(dim=1) / temperature
            relevance_loss = criterion(logits, label)
            positive_mask = label > 0.5
            if positive_mask.any():
                contrastive_features = torch.cat([query_emb[positive_mask], poi_emb[positive_mask]], dim=0)
                contrastive_labels = torch.cat([supcon_label[positive_mask], supcon_label[positive_mask]], dim=0)
                contrastive_loss = supervised_contrastive_loss(contrastive_features, contrastive_labels)
            else:
                contrastive_loss = torch.zeros((), device=label.device)
            loss = relevance_loss + 0.08 * contrastive_loss
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(f"  epoch {epoch + 1:03d}/{epochs} loss={history[-1]:.6f}")

    print("[6/8] Evaluating grouped holdout metrics")
    train_scores = predict_scores(model, train_dataset)
    test_scores = predict_scores(model, test_dataset)
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
            "type": "tfidf_two_tower_pytorch",
            "embedding_dim": 96,
            "epochs": epochs,
            "loss": "bce_relevance_plus_supervised_contrastive",
            "split": "grouped_by_sample_or_query",
            "purpose": "first neural representation baseline for agent POI matching",
        },
        "history": {"train_loss": history},
    }

    print(f"[7/8] Saving checkpoint to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vectorizer.vocabulary_,
            "metrics": metrics,
        },
        output_dir / "agent_two_tower_representation.pt",
    )
    print("[8/8] Writing metrics and training figure")
    (output_dir / "two_tower_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_training(history, metrics, output_dir)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    metrics = train(args.input, args.output_dir, args.epochs)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
