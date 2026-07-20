"""run_all - orchestrator that respects the scree go/no-go gate.

Order: assemble (1) -> delta (2) -> reduce (3) == SCREE GATE.
  - If run.stop_at_gate is true (default): stop after module 3 so the
    explained-variance curve can be inspected before committing to clustering.
  - Else the gate routes:
      PROCEED -> cluster (4) -> stability (5) -> qeeg (6) -> phenotype_stats (7)
                 -> report (8)
      STOP    -> graded_score (rank-1 fallback)

The gate is load-bearing (invariant 5): a rank-1 delta space never gets
silently clustered.

Run:  python -m analysis.run_all --config analysis/config.yaml
      python -m analysis.run_all --config analysis/config.yaml --from reduce
"""
from __future__ import annotations

import argparse
import os
import sys

from .config import Config, load_config, add_arg
from . import invariants
from . import assemble, delta, reduce as reduce_mod


STEPS_PRE_GATE = ["assemble", "delta", "reduce"]


def _ensure_repo_on_path(config: Config):
    repo_root = config.path("repo_root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _clinical_available(config: Config) -> bool:
    p = config.path("paths", "clinical_csv")
    return bool(p) and os.path.exists(p)


def _raw_eeg_available(config: Config) -> bool:
    p = config.path("paths", "raw_eeg_dir")
    if not p or not os.path.exists(p):
        return False
    if os.path.isdir(p):
        return len(os.listdir(p)) > 0
    return True                                  # a single HDF5 file


def _resolve(config: Config, key: str, auto_default: bool) -> bool:
    """run.<key> may be true/false to force, or null/absent to auto-detect."""
    val = config.get("run", {}).get(key, None)
    return auto_default if val is None else bool(val)


def run(config: Config, start_from: str = "assemble") -> dict:
    _ensure_repo_on_path(config)
    order = STEPS_PRE_GATE
    begin = order.index(start_from) if start_from in order else 0

    if begin <= 0:
        print("\n=== Module 1: assemble ===")
        assemble.main(config)
    if begin <= 1:
        print("\n=== Module 2: delta ===")
        delta.main(config)
    print("\n=== Module 3: reduce (SCREE GATE) ===")
    gate = reduce_mod.main(config)

    if bool(config["run"]["stop_at_gate"]):
        print("\n" + "=" * 68)
        print("HALTED AT SCREE GATE (run.stop_at_gate = true).")
        print(f"  decision : {gate['route'].upper()}")
        print(f"  rationale: {gate['rationale']}")
        print(f"  inspect  : {config.out('scree.png')}")
        print("  Set run.stop_at_gate=false (and implement modules 4-8) to "
              "continue automatically.")
        print("=" * 68)
        return gate

    # Gate routing.
    if invariants.should_proceed(gate):
        print("\n[run_all] gate: PROCEED -> clustering branch")
        from . import cluster, stability, qeeg, phenotype_stats, report, outlier_qc

        print("\n=== Outlier QC (diagnostic) ===")
        outlier_qc.main(config)
        print("\n=== Module 4: cluster ===")
        cluster.main(config)
        print("\n=== Module 5: stability ===")
        stability.main(config)

        # Module 6 (QEEG/FC): needs raw EEG. Auto-skip if unavailable.
        run_qeeg = _resolve(config, "run_qeeg", _raw_eeg_available(config))
        qeeg_ok = False
        if run_qeeg:
            print("\n=== Module 6: qeeg (functional connectivity) ===")
            try:
                qeeg.main(config)
                qeeg_ok = True
            except Exception as e:  # noqa: BLE001
                print(f"[run_all] module 6 (qeeg) failed/skipped: {type(e).__name__}: {e}")
        else:
            print("\n[run_all] skipping module 6 (qeeg): no raw EEG at "
                  f"{config.path('paths', 'raw_eeg_dir')}")

        # Module 7 (phenotype stats): needs the clinical covariates AND QEEG features.
        run_pheno = _resolve(config, "run_phenotype", _clinical_available(config))
        if run_pheno and qeeg_ok:
            print("\n=== Module 7: phenotype_stats ===")
            phenotype_stats.main(config)
        else:
            why = "no clinical table" if not run_pheno else "QEEG features not produced"
            print(f"\n[run_all] skipping module 7 (phenotype_stats): {why}. "
                  "The covariate-adjusted comparison needs both the clinical "
                  "covariates and QEEG features.")

        print("\n=== Module 8: report ===")
        report.main(config)
    else:
        print("\n[run_all] gate: STOP -> graded_score branch")
        from . import graded_score
        graded_score.main(config)
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EEG trajectory pipeline (gated)")
    add_arg(parser)
    parser.add_argument("--from", dest="start_from", default="assemble",
                        choices=STEPS_PRE_GATE, help="Resume from this step")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA device index (cuda:N) for module 2 training; "
                             "overrides config delta.device_index")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.raw.setdefault("delta", {})["device_index"] = args.device
    run(cfg, start_from=args.start_from)
