"""Fuzzy entity matching utilities for automated evaluation."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from rapidfuzz import fuzz


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalized(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def postal(value: object) -> str:
    return re.sub(r"\s+", "", clean(value)).casefold()


def domain(value: object) -> str:
    host = urlparse(clean(value)).netloc.casefold()
    return host.removeprefix("www.")


def entity_match_score(prediction: dict, target: dict) -> dict:
    name_available = bool(clean(target.get("name")) and clean(prediction.get("name")))
    address_available = bool(clean(target.get("address")) and clean(prediction.get("address")))
    city_available = bool(clean(target.get("city")) and clean(prediction.get("city")))
    name_score = fuzz.token_set_ratio(clean(prediction.get("name")), clean(target.get("name"))) / 100 if name_available else 0.0
    address_score = fuzz.token_set_ratio(clean(prediction.get("address")), clean(target.get("address"))) / 100 if address_available else 0.0
    city_score = fuzz.token_set_ratio(clean(prediction.get("city")), clean(target.get("city"))) / 100 if city_available else 0.0
    target_postal = postal(target.get("postal_code"))
    prediction_postal = postal(prediction.get("postal_code"))
    postal_available = bool(target_postal and prediction_postal)
    postal_score = 1.0 if postal_available and target_postal == prediction_postal else 0.0
    target_domain = domain(target.get("website"))
    prediction_domain = domain(prediction.get("website"))
    website_available = bool(target_domain and prediction_domain)
    website_score = 1.0 if website_available and target_domain == prediction_domain else 0.0
    components = ((name_score, 0.40, name_available), (address_score, 0.30, address_available), (city_score, 0.15, city_available), (postal_score, 0.10, postal_available), (website_score, 0.05, website_available))
    available_weight = sum(weight for _, weight, available in components if available)
    weighted = sum(score * weight for score, weight, available in components if available) / available_weight if available_weight else 0.0
    return {
        "score": round(weighted, 4),
        "name_score": round(name_score, 4),
        "address_score": round(address_score, 4),
        "city_score": round(city_score, 4),
        "postal_score": postal_score,
        "website_score": website_score,
        "supported": weighted >= 0.72 and name_score >= 0.60 and city_score >= 0.70,
    }
