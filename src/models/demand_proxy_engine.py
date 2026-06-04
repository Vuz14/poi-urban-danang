"""Demand proxy scoring for business-location MVP.

The output is not real footfall. It is an explainable proxy from POI density,
review/rating quality, complementary POIs, and later first-party app behavior.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

from .semantic_reranker import clamp01, keyword_score, normalize_text, rating_score


@dataclass
class DemandProxy:
    score: float
    signals: dict[str, float]
    explanation: str


def review_signal(review_count: object) -> float:
    try:
        count = float(review_count or 0)
    except (TypeError, ValueError):
        count = 0
    return clamp01(math.log10(count + 1) / 3)


def category_density(pois: Iterable[Mapping[str, object]], concept: str) -> float:
    pois = list(pois)
    if not pois:
        return 0.0
    concept_norm = normalize_text(concept)
    hits = 0
    for poi in pois:
        category = normalize_text(poi.get("category") or poi.get("Category") or "")
        text = normalize_text(poi.get("LLM_Input_Text") or poi.get("Aggregated_Reviews") or "")
        if keyword_score(concept_norm, f"{category} {text}") > 0:
            hits += 1
    return clamp01(hits / max(len(pois), 1))


def complementary_density(pois: Iterable[Mapping[str, object]]) -> float:
    categories = Counter(
        normalize_text(poi.get("category") or poi.get("Category") or "unknown")
        for poi in pois
    )
    diversity = len(categories)
    total = sum(categories.values())
    return clamp01((math.log10(total + 1) / 2) * 0.65 + (min(diversity, 12) / 12) * 0.35)


def compute_demand_proxy(
    concept: str,
    pois: Iterable[Mapping[str, object]],
    app_behavior_score: float = 0.0,
) -> DemandProxy:
    pois = list(pois)
    if not pois:
        return DemandProxy(0.0, {}, "Khong co POI trong khu vuc de tinh proxy.")

    ratings = [rating_score(poi.get("rating") or poi.get("Overall Rating")) for poi in pois]
    reviews = [
        review_signal(poi.get("review_count") or poi.get("Total_Reviews_Scraped"))
        for poi in pois
    ]
    signals = {
        "category_density": category_density(pois, concept),
        "complementary_density": complementary_density(pois),
        "rating_quality": sum(ratings) / len(ratings),
        "review_signal": sum(reviews) / len(reviews),
        "app_behavior": clamp01(app_behavior_score),
    }
    score = clamp01(
        signals["category_density"] * 0.28
        + signals["complementary_density"] * 0.22
        + signals["rating_quality"] * 0.18
        + signals["review_signal"] * 0.17
        + signals["app_behavior"] * 0.15
    )
    return DemandProxy(
        score=score,
        signals=signals,
        explanation=(
            "Demand proxy duoc uoc luong tu mat do danh muc, POI bo tro, "
            "chat luong rating/review va hanh vi app neu co."
        ),
    )
