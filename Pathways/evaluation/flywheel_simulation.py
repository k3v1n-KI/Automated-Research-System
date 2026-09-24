"""Deterministic flywheel simulation for Study 3.

This is intentionally lightweight: it models a resource directory whose fields age
over time, then compares a static control to a verification flywheel that can
refresh stale records at a configurable rate.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from statistics import fmean, pstdev
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _freshness_score(age_days: int, stale_after_days: int) -> float:
    if stale_after_days <= 0:
        return 1.0
    if age_days <= 0:
        return 1.0
    return max(0.0, 1.0 - (age_days / stale_after_days))


def simulate_flywheel(
    timepoints: tuple[int, ...] = (0, 30, 90),
    verification_rate: float = 0.5,
    stale_after_days: int = 120,
    seed: int = 0,
) -> dict[str, Any]:
    """Simulate two conditions over a small deterministic resource set.

    - static_control: resources decay without maintenance.
    - flywheel: a fraction of stale records is refreshed by verification events.
    """
    rng = random.Random(seed)
    resources = [
        {"id": f"r-{index}", "age_days": 0, "verified": True}
        for index in range(1, 11)
    ]

    static_control: dict[int, dict[str, float]] = {}
    flywheel: dict[int, dict[str, float]] = {}

    for t in timepoints:
        static = []
        fly = []
        for resource in resources:
            age = max(0, t)
            static.append(_freshness_score(age, stale_after_days))

            # flywheel: some resources are verified and refreshed before they become stale
            if resource["verified"] and t > 0 and rng.random() < verification_rate:
                fly.append(1.0)
            else:
                fly.append(_freshness_score(age, stale_after_days))
        static_control[t] = {
            "fresh_rate": sum(score >= 0.75 for score in static) / len(static),
            "stale_rate": sum(score < 0.25 for score in static) / len(static),
            "mean_freshness": sum(static) / len(static),
        }
        flywheel[t] = {
            "fresh_rate": sum(score >= 0.75 for score in fly) / len(fly),
            "stale_rate": sum(score < 0.25 for score in fly) / len(fly),
            "mean_freshness": sum(fly) / len(fly),
        }

    summary = {
        "static_fresh_rate": static_control[max(timepoints)]["fresh_rate"],
        "flywheel_fresh_rate": flywheel[max(timepoints)]["fresh_rate"],
        "static_stale_rate": static_control[max(timepoints)]["stale_rate"],
        "flywheel_stale_rate": flywheel[max(timepoints)]["stale_rate"],
    }

    return {
        "schema_version": 1,
        "evaluation_type": "flywheel_simulation",
        "timepoints": list(timepoints),
        "conditions": ["static_control", "flywheel"],
        "static_control": static_control,
        "flywheel": flywheel,
        "summary": summary,
    }


def sweep_flywheel(
    timepoints: tuple[int, ...] = (0, 30, 90),
    verification_rates: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    seeds: tuple[int, ...] = tuple(range(10)),
    stale_after_days: int = 90,
) -> dict[str, Any]:
    """Aggregate simulated flywheel outcomes across participation and seeds."""
    if not timepoints:
        raise ValueError("At least one timepoint is required")
    final_timepoint = max(timepoints)
    scenarios: dict[str, dict[str, float | int]] = {}
    for verification_rate in verification_rates:
        reports = [
            simulate_flywheel(
                timepoints=timepoints,
                verification_rate=verification_rate,
                stale_after_days=stale_after_days,
                seed=seed,
            )
            for seed in seeds
        ]
        flywheel_fresh = [report["flywheel"][final_timepoint]["fresh_rate"] for report in reports]
        flywheel_stale = [report["flywheel"][final_timepoint]["stale_rate"] for report in reports]
        static_fresh = [report["static_control"][final_timepoint]["fresh_rate"] for report in reports]
        scenarios[str(verification_rate)] = {
            "seed_count": len(reports),
            "t90_flywheel_fresh_rate_mean": fmean(flywheel_fresh) if flywheel_fresh else 0.0,
            "t90_flywheel_fresh_rate_std": pstdev(flywheel_fresh) if len(flywheel_fresh) > 1 else 0.0,
            "t90_flywheel_stale_rate_mean": fmean(flywheel_stale) if flywheel_stale else 0.0,
            "t90_static_fresh_rate_mean": fmean(static_fresh) if static_fresh else 0.0,
        }
    return {
        "schema_version": 1,
        "evaluation_type": "flywheel_participation_sweep",
        "timepoints": list(timepoints),
        "final_timepoint": final_timepoint,
        "verification_rates": list(verification_rates),
        "seed_count": len(seeds),
        "stale_after_days": stale_after_days,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timepoints", nargs="*", type=int, default=[0, 30, 90])
    parser.add_argument("--verification-rate", type=float, default=0.5)
    parser.add_argument("--stale-after-days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--verification-rates", nargs="*", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.sweep:
        report = sweep_flywheel(
            timepoints=tuple(args.timepoints),
            verification_rates=tuple(args.verification_rates),
            seeds=tuple(args.seeds),
            stale_after_days=args.stale_after_days,
        )
    else:
        report = simulate_flywheel(
            timepoints=tuple(args.timepoints),
            verification_rate=args.verification_rate,
            stale_after_days=args.stale_after_days,
            seed=args.seed,
        )

    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
