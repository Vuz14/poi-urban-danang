"""Lightweight semantic reranking utilities for Danang UrbanAgent AI.

This module is intentionally dependency-light so it can be reused by scripts,
notebooks, or exported into the backend later.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping


def normalize_text(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def tokenize(value: object) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", normalize_text(value)) if len(token) >= 2]


def keyword_score(query: str, document: str) -> float:
    tokens = sorted(set(tokenize(query)))
    if not tokens:
        return 0.0
    doc = normalize_text(document)
    return sum(1 for token in tokens if token in doc) / len(tokens)


def clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def rating_score(rating: object) -> float:
    try:
        value = float(rating)
    except (TypeError, ValueError):
        return 0.35
    if value > 5:
        return clamp01(value / 10)
    return clamp01(value / 5)


@dataclass
class RerankWeights:
    semantic: float = 0.42
    category: float = 0.24
    rating: float = 0.16
    distance: float = 0.10
    opening: float = 0.08


def category_match_score(query: str, category: str) -> float:
    query_norm = normalize_text(query)
    category_norm = normalize_text(category)
    category_groups = {
        "cafe": ["cafe", "ca phe", "coffee", "dessert", "tra sua"],
        "seafood": ["hai san", "seafood"],
        "food": ["quan an", "an vat", "via he", "food"],
        "pub": ["quan nhau", "nhau", "beer", "bar"],
        "travel": ["du lich", "tham quan", "check in", "bien"],
    }
    for terms in category_groups.values():
        query_hit = any(term in query_norm for term in terms)
        category_hit = any(term in category_norm for term in terms)
        if query_hit and category_hit:
            return 1.0
    return 0.45 if keyword_score(query_norm, category_norm) else 0.12


def rerank_pois(
    query: str,
    pois: Iterable[Mapping[str, object]],
    weights: RerankWeights | None = None,
) -> list[dict[str, object]]:
    weights = weights or RerankWeights()
    results: list[dict[str, object]] = []
    for poi in pois:
        text = " ".join(
            str(poi.get(key, ""))
            for key in ("name", "Restaurant Name", "category", "Category", "LLM_Input_Text", "Aggregated_Reviews")
        )
        semantic = keyword_score(query, text)
        category = category_match_score(query, str(poi.get("category") or poi.get("Category") or ""))
        rating = rating_score(poi.get("rating") or poi.get("Overall Rating"))
        distance = float(poi.get("distance_score", 0.5) or 0.5)
        opening = float(poi.get("opening_score", 0.7) or 0.7)
        score = clamp01(
            semantic * weights.semantic
            + category * weights.category
            + rating * weights.rating
            + distance * weights.distance
            + opening * weights.opening
        )
        enriched = dict(poi)
        enriched["rerank_score"] = score
        enriched["rerank_signals"] = {
            "semantic": semantic,
            "category": category,
            "rating": rating,
            "distance": distance,
            "opening": opening,
        }
        results.append(enriched)
    return sorted(results, key=lambda item: item["rerank_score"], reverse=True)
