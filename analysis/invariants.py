"""The five correctness invariants, as pure, dependency-light functions.

These are the single source of truth for the guarantees in
ANALYSIS_PIPELINE_SPEC.md ("Correctness invariants"). Every module calls into
this module at the point where an invariant must hold, and ``analysis/tests``
exercises these same functions directly. Keeping them here (numpy/pandas only,
no torch/statsmodels) means the invariants are testable without the heavy
runtime stack.

Invariants
----------
1. No patient split across folds/groups anywhere.
2. Deltas consistently earlier -> later oriented.
3. Clustering unit = progression, never segment.
4. Phenotype stats carry the confound covariates + a patient random effect.
5. The scree gate can halt the pipeline before clustering.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Required confound covariates for every phenotype comparison (invariant 4).
REQUIRED_CONFOUNDS = ("dt", "baseline_severity", "APOE4", "ARIA", "age")


class InvariantError(AssertionError):
    """Raised when a pipeline correctness invariant is violated."""


# ---------------------------------------------------------------------------
# Invariant 1 - patient groups are disjoint across folds
# ---------------------------------------------------------------------------
def assert_disjoint_groups(df: pd.DataFrame,
                           patient_col: str = "patient_id",
                           group_col: str = "fold") -> None:
    """Each patient must live in exactly one group/fold.

    Guards training folds, the PCA fold assignment, and stability resamples so
    no patient's progressions leak across a subject-wise split.
    """
    per_patient = df.groupby(patient_col)[group_col].nunique()
    offenders = per_patient[per_patient > 1]
    if len(offenders) > 0:
        raise InvariantError(
            "Invariant 1 violated: patient(s) span multiple '%s' groups: %s"
            % (group_col, offenders.to_dict()))


def assert_resample_groups(train_patients: Iterable,
                           held_patients: Iterable) -> None:
    """No patient appears in both sides of a resample/train-test split."""
    overlap = set(train_patients) & set(held_patients)
    if overlap:
        raise InvariantError(
            "Invariant 1 violated: %d patient(s) in both partitions: %s"
            % (len(overlap), sorted(overlap)[:10]))


# ---------------------------------------------------------------------------
# Invariant 2 - deltas are earlier -> later oriented
# ---------------------------------------------------------------------------
def oriented_pair(date_a, emb_a, date_b, emb_b):
    """Return (earlier_embedding, later_embedding) ordered by session date.

    The training dataset randomly flips time order for augmentation; delta
    computation must NOT inherit that. This helper forces a canonical
    earlier -> later orientation from the actual session dates.
    """
    if pd.Timestamp(date_a) <= pd.Timestamp(date_b):
        return emb_a, emb_b
    return emb_b, emb_a


def assert_earlier_to_later(before_dates: Sequence, after_dates: Sequence) -> None:
    """Every progression's 'after' date must be strictly later than 'before'.

    Run at delta aggregation; a single flipped pair would let PC1 split on the
    orientation artifact instead of real change.
    """
    b = pd.to_datetime(pd.Series(before_dates))
    a = pd.to_datetime(pd.Series(after_dates))
    bad = np.where(~(a.values > b.values))[0]
    if len(bad) > 0:
        raise InvariantError(
            "Invariant 2 violated: %d progression(s) not strictly earlier->later "
            "(first offending row indices: %s)" % (len(bad), bad[:10].tolist()))


# ---------------------------------------------------------------------------
# Invariant 3 - the analysis unit is the progression, not the segment
# ---------------------------------------------------------------------------
def assert_progression_unit(n_rows: int, progression_ids: Sequence) -> None:
    """The delta / feature matrix must have exactly one row per progression."""
    pids = list(progression_ids)
    n_unique = len(set(pids))
    if not (n_rows == len(pids) == n_unique):
        raise InvariantError(
            "Invariant 3 violated: expected one row per unique progression; "
            "got n_rows=%d, n_ids=%d, n_unique_ids=%d (segments not collapsed?)"
            % (n_rows, len(pids), n_unique))


# ---------------------------------------------------------------------------
# Invariant 4 - phenotype model carries confounds + patient random effect
# ---------------------------------------------------------------------------
def validate_phenotype_model_spec(covariates: Iterable[str],
                                  has_patient_effect: bool) -> None:
    """A cluster/phenotype comparison must adjust for the confounds and account
    for duplicate patients via a random effect or cluster-robust SE."""
    covset = set(covariates)
    missing = [c for c in REQUIRED_CONFOUNDS if c not in covset]
    if missing:
        raise InvariantError(
            "Invariant 4 violated: model missing required confounds: %s" % missing)
    if not has_patient_effect:
        raise InvariantError(
            "Invariant 4 violated: model must include a patient random effect "
            "(or cluster-robust SE) for duplicate patients.")


# ---------------------------------------------------------------------------
# Invariant 5 - the scree gate can halt before clustering
# ---------------------------------------------------------------------------
def broken_stick(p: int) -> np.ndarray:
    """Broken-stick expected proportion of variance for p ordered components."""
    idx = np.arange(1, p + 1)
    return np.array([np.sum(1.0 / idx[k - 1:]) / p for k in idx])


def gate_decision(explained_variance_ratio: Sequence[float],
                  min_components_above_floor: int = 2,
                  pc1_dominance_threshold: float = 0.80,
                  noise_floor: float = 0.05,
                  use_broken_stick: bool = True) -> dict:
    """Decide whether to PROCEED to clustering or STOP and route to a graded score.

    A component counts as "signal" if it rises above the noise floor. With the
    broken-stick model the floor is component-specific (its null expectation);
    otherwise a flat ``noise_floor`` is used. If fewer than
    ``min_components_above_floor`` components are signal -- i.e. the delta space
    is effectively rank-1 -- the clustering branch is halted and the run is
    routed to ``graded_score.py``.
    """
    evr = np.asarray(list(explained_variance_ratio), dtype=float)
    p = len(evr)
    if p == 0:
        raise InvariantError("Invariant 5: empty explained-variance vector.")

    bstick = broken_stick(p) if use_broken_stick else np.full(p, noise_floor)
    floor = np.maximum(bstick, noise_floor) if use_broken_stick else bstick
    signal_mask = evr > floor
    n_signal = int(np.sum(signal_mask))

    pc1 = float(evr[0])
    pc2 = float(evr[1]) if p > 1 else 0.0

    rank1_by_count = n_signal < min_components_above_floor
    rank1_by_dominance = (pc1 >= pc1_dominance_threshold) and (pc2 < noise_floor)
    proceed = not (rank1_by_count or rank1_by_dominance)

    if proceed:
        rationale = (f"{n_signal} component(s) above floor (>= "
                     f"{min_components_above_floor}); PC1={pc1:.3f}, PC2={pc2:.3f}. "
                     f"Delta space has structure -> proceed to clustering.")
    else:
        reasons = []
        if rank1_by_count:
            reasons.append(f"only {n_signal} component(s) above floor "
                           f"(< {min_components_above_floor})")
        if rank1_by_dominance:
            reasons.append(f"PC1={pc1:.3f} >= {pc1_dominance_threshold} with "
                           f"PC2={pc2:.3f} < {noise_floor}")
        rationale = ("Rank-1 delta space (" + "; ".join(reasons) +
                     ") -> HALT clustering, route to graded_score.py.")

    return {
        "proceed": bool(proceed),
        "route": "clustering" if proceed else "graded_score",
        "n_signal_components": n_signal,
        "signal_mask": signal_mask.tolist(),
        "pc1_ratio": pc1,
        "pc2_ratio": pc2,
        "noise_floor": noise_floor,
        "broken_stick": bstick.tolist(),
        "used_broken_stick": bool(use_broken_stick),
        "explained_variance_ratio": evr.tolist(),
        "rationale": rationale,
    }


def should_proceed(gate: dict) -> bool:
    """run_all consults this: the gate can and must be able to halt the run."""
    return bool(gate.get("proceed", False))
