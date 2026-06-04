"""Business location scoring prototype for Danang UrbanAgent AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .demand_proxy_engine import compute_demand_proxy
from .semantic_reranker import clamp01, category_match_score, keyword_score, normalize_text


@dataclass
class BusinessLocationScore:
    score: float
    signals: dict[str, float]
    reason: str
    warnings: list[str]


def competition_penalty(concept: str, pois: Iterable[Mapping[str, object]]) -> float:
    pois = list(pois)
    if not pois:
        return 0.0
    direct = 0
    for poi in pois:
        category = str(poi.get("category") or poi.get("Category") or "")
        if category_match_score(concept, category) > 0.7:
            direct += 1
    return clamp01(direct / 12)


def concept_fit(concept: str, pois: Iterable[Mapping[str, object]]) -> float:
    pois = list(pois)
    if not pois:
        return 0.0
    concept_norm = normalize_text(concept)
    hits = 0
    for poi in pois:
        text = " ".join(
            str(poi.get(key, ""))
            for key in ("name", "Restaurant Name", "category", "Category", "LLM_Input_Text", "Aggregated_Reviews")
        )
        if keyword_score(concept_norm, text) > 0:
            hits += 1
    return clamp01(hits / max(len(pois), 1))


def score_business_location(
    concept: str,
    pois_in_area: Iterable[Mapping[str, object]],
    accessibility_score: float = 0.6,
) -> BusinessLocationScore:
    pois = list(pois_in_area)
    demand = compute_demand_proxy(concept, pois)
    competition = competition_penalty(concept, pois)
    fit = concept_fit(concept, pois)
    accessibility = clamp01(accessibility_score)
    opportunity = clamp01(
        demand.score * 0.34
        + demand.signals.get("complementary_density", 0) * 0.18
        + accessibility * 0.20
        + fit * 0.20
        - competition * 0.12
    )
    warnings: list[str] = []
    if competition > 0.65:
        warnings.append("Canh tranh truc tiep cao, can khac biet hoa concept.")
    if demand.score < 0.25:
        warnings.append("Tin hieu nhu cau con yeu voi du lieu hien co.")
    return BusinessLocationScore(
        score=opportunity,
        signals={
            "demand_proxy": demand.score,
            "competition_penalty": competition,
            "concept_fit": fit,
            "accessibility": accessibility,
            **demand.signals,
        },
        reason=(
            "Diem co hoi ket hop demand proxy, POI bo tro, kha nang tiep can, "
            "do hop concept va muc do canh tranh."
        ),
        warnings=warnings,
    )
