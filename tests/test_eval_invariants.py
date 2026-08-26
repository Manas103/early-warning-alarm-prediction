"""Reproducibility invariants for the eval harness output: a fixed seed
range must always regenerate byte-identical run data and the same ground
truth labels, so the catch/false-alarm/lead-time counts in docs/eval_output.txt
are not an artifact of a lucky, unreproducible draw."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from simulate import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_eval_seed_range_reproducible():
    a = make_dataset(5, 5, seed_offset=900000)
    b = make_dataset(5, 5, seed_offset=900000)
    for ra, rb in zip(a, b):
        assert ra.seed == rb.seed
        assert ra.is_upset == rb.is_upset
        assert ra.alarm_minute == rb.alarm_minute
        assert (ra.channels == rb.channels).all()


def test_committed_eval_summary_is_internally_consistent():
    summary_path = ROOT / "docs" / "eval_summary.json"
    if not summary_path.exists():
        import pytest
        pytest.skip("docs/eval_summary.json not generated in this checkout")
    summary = json.loads(summary_path.read_text())
    assert summary["n_upset"] == 40
    assert summary["n_healthy"] == 40
    assert 0 <= summary["n_caught"] <= summary["n_upset"]
    assert 0 <= summary["n_false_alarms"] <= summary["n_healthy"]


def test_committed_manifest_ground_truth_matches_regenerated_runs():
    manifest_path = ROOT / "data" / "eval_runs" / "manifest.json"
    if not manifest_path.exists():
        import pytest
        pytest.skip("data/eval_runs/manifest.json not present in this checkout (gitignored, regenerable)")
    manifest = json.loads(manifest_path.read_text())
    from simulate import generate_run
    for r in manifest["runs"][:10]:  # spot check, full set is slower
        regenerated = generate_run(seed=r["seed"], is_upset=r["is_upset"])
        assert regenerated.alarm_minute == r["alarm_minute"]
        assert regenerated.t0 == r["t0"]
