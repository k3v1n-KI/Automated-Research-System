"""Small Google Places Text Search adapter for Pathways corroboration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


_dotenv_path = Path(__file__).resolve().parent / "searxng" / ".env"
if _dotenv_path.exists():
    load_dotenv(_dotenv_path)


def search_places(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        return []
    response = requests.post(
        "https://places.googleapis.com/v1/places:searchText",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": (
                "places.id,places.displayName,places.formattedAddress,"
                "places.nationalPhoneNumber,places.websiteUri,places.googleMapsUri,"
                "places.currentOpeningHours"
            ),
        },
        json={"textQuery": query, "maxResultCount": min(max_results, 20)},
        timeout=30,
    )
    response.raise_for_status()
    records = []
    for place in response.json().get("places", []):
        name = place.get("displayName", {}).get("text", "")
        address = place.get("formattedAddress", "")
        postal_match = re.search(r"\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b", address, re.IGNORECASE)
        postal_code = postal_match.group(0).upper() if postal_match else ""
        address_parts = [part.strip() for part in address.split(",")]
        city = address_parts[-3] if len(address_parts) >= 3 else ""
        phone = place.get("nationalPhoneNumber", "")
        hours = place.get("currentOpeningHours", {}).get("weekdayDescriptions", [])
        hours_text = " | ".join(hours)
        evidence = f"Google Places lists {name} at {address}."
        if phone:
            evidence += f" Phone: {phone}."
        if hours_text:
            evidence += f" Hours: {hours_text}."
        records.append(
            {
                "name": name,
                "address": address,
                "city": city,
                "postal_code": postal_code,
                "phone": phone,
                "website": place.get("websiteUri", ""),
                "hours": hours_text,
                "source_url": place.get("googleMapsUri", ""),
                "evidence": evidence,
                "place_id": place.get("id", ""),
            }
        )
    return records