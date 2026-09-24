from datetime import datetime, timedelta, timezone

from evaluation.flywheel_simulation import simulate_flywheel, sweep_flywheel


def test_study3_flywheel_simulation_tracks_freshness_over_time():
    report = simulate_flywheel(
        timepoints=(0, 30, 90),
        verification_rate=0.5,
        stale_after_days=120,
        seed=7,
    )

    assert report["evaluation_type"] == "flywheel_simulation"
    assert report["timepoints"] == [0, 30, 90]
    assert set(report["static_control"]) == {0, 30, 90}
    assert set(report["flywheel"]) == {0, 30, 90}
    assert report["flywheel"][90]["fresh_rate"] >= report["static_control"][90]["fresh_rate"]
    assert report["flywheel"][90]["stale_rate"] <= report["static_control"][90]["stale_rate"]


def test_study3_flywheel_report_contains_summary_metrics():
    report = simulate_flywheel(timepoints=(0, 30, 90), verification_rate=1.0, stale_after_days=30, seed=9)

    assert 0.0 <= report["summary"]["flywheel_fresh_rate"] <= 1.0
    assert 0.0 <= report["summary"]["static_fresh_rate"] <= 1.0
    assert "timepoints" in report
    assert "conditions" in report


def test_study3_sweep_aggregates_participation_rates_and_seeds():
    report = sweep_flywheel(
        timepoints=(0, 30, 90),
        verification_rates=(0.0, 0.5, 1.0),
        seeds=(1, 2, 3),
        stale_after_days=90,
    )

    assert report["evaluation_type"] == "flywheel_participation_sweep"
    assert report["verification_rates"] == [0.0, 0.5, 1.0]
    assert report["seed_count"] == 3
    assert report["scenarios"]["1.0"]["t90_flywheel_fresh_rate_mean"] >= report["scenarios"]["0.0"]["t90_flywheel_fresh_rate_mean"]
