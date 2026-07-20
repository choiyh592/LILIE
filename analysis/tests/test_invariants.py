"""Tests for the five correctness invariants (ANALYSIS_PIPELINE_SPEC.md).

Pure-logic tests over the enforcement functions in ``analysis.invariants`` with
synthetic data - no torch / statsmodels needed. Runnable under pytest OR as a
plain script (``python analysis/tests/test_invariants.py``).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# allow running as a plain script from the repo root
sys.path.insert(0, ".")

from analysis import invariants
from analysis.phenotype_stats import covariate_spec
from analysis.config import Config


def _expect_raises(fn, exc=invariants.InvariantError):
    try:
        fn()
    except exc:
        return True
    raise AssertionError(f"expected {exc.__name__} but none was raised")


# --- Invariant 1: no patient split across folds -----------------------------
def test_invariant1_disjoint_groups_ok():
    df = pd.DataFrame({"patient_id": [1, 1, 2, 3, 3], "fold": [1, 1, 2, 3, 3]})
    invariants.assert_disjoint_groups(df)  # must not raise


def test_invariant1_disjoint_groups_violation():
    df = pd.DataFrame({"patient_id": [1, 1, 2], "fold": [1, 2, 2]})  # pt 1 in 2 folds
    _expect_raises(lambda: invariants.assert_disjoint_groups(df))


def test_invariant1_resample_overlap_violation():
    _expect_raises(lambda: invariants.assert_resample_groups([1, 2, 3], [3, 4]))
    invariants.assert_resample_groups([1, 2], [3, 4])  # disjoint ok


# --- Invariant 2: deltas earlier -> later -----------------------------------
def test_invariant2_orientation_forced_from_dates():
    ea, eb = np.array([1.0, 1.0]), np.array([3.0, 3.0])
    # feed dates in reversed order; helper must still return earlier, later
    early, late = invariants.oriented_pair("2021-06-01", eb, "2020-01-01", ea)
    assert np.allclose(early, ea) and np.allclose(late, eb)
    # delta earlier->later is positive here
    assert np.all((late - early) > 0)


def test_invariant2_flipped_pair_detected():
    before = ["2020-01-01", "2020-06-01", "2021-01-01"]
    after = ["2020-06-01", "2020-01-01", "2021-06-01"]  # row 1 flipped
    _expect_raises(lambda: invariants.assert_earlier_to_later(before, after))
    # all-consistent passes
    invariants.assert_earlier_to_later(["2020-01-01"], ["2020-02-01"])


# --- Invariant 3: clustering unit = progression -----------------------------
def test_invariant3_progression_unit_ok():
    ids = ["p1", "p2", "p3"]
    invariants.assert_progression_unit(3, ids)


def test_invariant3_segment_rows_detected():
    # 6 segment rows but only 3 progressions -> must fail (not collapsed)
    ids = ["p1", "p1", "p2", "p2", "p3", "p3"]
    _expect_raises(lambda: invariants.assert_progression_unit(6, ids))


# --- Invariant 4: phenotype model carries confounds + patient effect --------
def test_invariant4_full_spec_ok():
    invariants.validate_phenotype_model_spec(
        ["dt", "baseline_severity", "APOE4", "ARIA", "age"], has_patient_effect=True)


def test_invariant4_missing_confound_detected():
    _expect_raises(lambda: invariants.validate_phenotype_model_spec(
        ["dt", "APOE4", "ARIA", "age"], has_patient_effect=True))  # no baseline_severity


def test_invariant4_missing_patient_effect_detected():
    _expect_raises(lambda: invariants.validate_phenotype_model_spec(
        ["dt", "baseline_severity", "APOE4", "ARIA", "age"], has_patient_effect=False))


def test_invariant4_config_spec_validates():
    cfg = Config(raw={"phenotype_stats": {
        "covariates": ["dt", "baseline_severity", "APOE4", "ARIA", "age"],
        "patient_random_effect": True, "model": "mixedlm",
        "primary_features": ["a", "b"], "fdr_alpha": 0.05}}, config_dir=".")
    spec = covariate_spec(cfg)
    assert spec["groups"] == "patient_id"


# --- Invariant 5: scree gate can halt ---------------------------------------
def test_invariant5_rank1_halts():
    evr = [0.92, 0.03, 0.02, 0.015, 0.015]     # PC1 dominates, rest at floor
    gate = invariants.gate_decision(evr)
    assert gate["proceed"] is False
    assert gate["route"] == "graded_score"
    assert invariants.should_proceed(gate) is False


def test_invariant5_multicomponent_proceeds():
    evr = [0.40, 0.25, 0.15, 0.10, 0.10]       # several real components
    gate = invariants.gate_decision(evr)
    assert gate["proceed"] is True
    assert gate["route"] == "clustering"
    assert invariants.should_proceed(gate) is True


def test_invariant5_borderline_two_components():
    evr = [0.55, 0.30, 0.05, 0.05, 0.05]
    gate = invariants.gate_decision(evr, min_components_above_floor=2)
    assert gate["n_signal_components"] >= 2
    assert gate["proceed"] is True


# --- plain-script runner (pytest not required) ------------------------------
def _run():
    fns = {k: v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)}
    failed = 0
    for name, fn in fns.items():
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(_run())
