"""EEG Trajectory Phenotype analysis pipeline.

Ordered, independently runnable steps (modules 1-8) that turn precomputed
LaBraM segment embeddings + preprocessed EEG into covariate-adjusted
phenotype comparisons. See README.md for the module -> pre-registered-plan
mapping and ANALYSIS_PIPELINE_SPEC.md for the source brief.

Design rule: all five correctness invariants are enforced by the pure
functions in ``analysis.invariants`` and exercised by ``analysis/tests``.
Heavy / optional deps (torch, statsmodels) are imported lazily inside the
functions that need them, so importing this package for the invariant tests
never requires them.
"""

__all__ = ["invariants", "config", "io"]
