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

Downstream steps (run.downstream, list or comma-string):
  directional  - reliable-direction vMF + spherical phenotyping
  trajectory   - clusterability + direction x magnitude
  axis         - the full axis-analysis suite (geometry -> compass -> qeeg ->
                 controls -> axis_qeeg -> axis_slowing -> axis_aperiodic), each
                 stage guarded (skips with a reason if its inputs are absent)
  clustering   - legacy hard-clustering branch (kept, off by default)
  qc           - psd_diagnostic (line-noise / notch check)
  all          - directional + trajectory + axis (the one-shot for a full run)

Run:  python -m analysis.run_all --config analysis/config.yaml
      python -m analysis.run_all --config analysis/config.yaml --from reduce
For a complete new experiment set run.stop_at_gate=false and run.downstream=all.
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


def _downstream_steps(config: Config) -> set:
    """run.downstream may be a list, a comma-string, or a single value.
    Legacy: "both" == trajectory + clustering. "all" == directional+trajectory+axis."""
    ds = config.get("run", {}).get("downstream", "directional")
    if isinstance(ds, str):
        if ds == "both":
            return {"trajectory", "clustering"}
        steps = {s.strip() for s in ds.split(",") if s.strip()}
    else:
        steps = set(ds)
    if "all" in steps:
        steps |= {"directional", "trajectory", "axis"}
        steps.discard("all")
    return steps


def _have(config: Config, fname: str) -> bool:
    return os.path.exists(config.out(fname))


def _have_table(config: Config, base: str) -> bool:
    return (os.path.exists(config.out(base + ".parquet"))
            or os.path.exists(config.out(base + ".csv")))


def _control_available(config: Config) -> bool:
    for k in ("control_metadata_csv", "control_embeddings_npy"):
        p = config.get("paths", {}).get(k)
        if not p or not os.path.exists(config.path("paths", k) or ""):
            return False
    return os.path.isdir(config.out("_ckpt"))


def _step(label: str, fn) -> bool:
    """Run a stage; a missing-input SystemExit is a SKIP, any other error is
    logged and swallowed so one bad stage can't sink the whole run."""
    print(f"\n=== {label} ===")
    try:
        fn(); return True
    except SystemExit as e:
        print(f"[run_all] SKIP {label}: {e}"); return False
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"[run_all] FAILED {label}: {type(e).__name__}: {e}")
        traceback.print_exc(); return False


def _run_axis_suite(config: Config):
    """directional -> reliability -> controls -> geometry -> compass -> qeeg ->
    axis_qeeg/slowing/aperiodic. Controls run BEFORE the compass so control_deltas.npz
    exists and the compass can plot them. Each stage guarded."""
    from . import directional_phenotype, phenotype_geometry, phenotype_compass
    from . import qeeg as qeeg_mod

    if not _have(config, "directional_phenotype_labels.csv"):
        _step("Directional phenotyping (vMF + spherical)",
              lambda: directional_phenotype.main(config))

    # Decisive salvage test: is the delta DIRECTION reproducible within a session?
    if os.path.isdir(config.out("_ckpt")):
        from . import delta_reliability
        _step("Delta direction reliability (split-half)",
              lambda: delta_reliability.main(config))
    else:
        print("\n[run_all] SKIP delta reliability: no fold checkpoints (run module 2).")

    # Controls FIRST -> control_deltas.npz available to the compass (and everything else).
    if _control_available(config):
        from . import control_deltas, control_analysis
        if _step("Control deltas (apply treated models to untreated)",
                 lambda: control_deltas.main(config)):
            _step("Control analysis (projection + magnitude test)",
                  lambda: control_analysis.main(config))
    else:
        print("\n[run_all] SKIP control cohort: control metadata/embeddings or "
              "fold checkpoints not found.")

    _step("Phenotype geometry (auto-k + angle-null)",
          lambda: phenotype_geometry.main(config))
    _step("Phenotype compass (writes phenotype_axis.csv; controls now available)",
          lambda: phenotype_compass.main(config))

    if _resolve(config, "run_qeeg", _raw_eeg_available(config)):
        _step("Module 6: qeeg (connectivity + spectral)", lambda: qeeg_mod.main(config))
    else:
        print("\n[run_all] SKIP qeeg: no raw EEG at "
              f"{config.path('paths', 'raw_eeg_dir')}.")

    if _have_table(config, "qeeg_connectivity") and _have(config, "phenotype_axis.csv"):
        from . import axis_qeeg, axis_slowing_test, axis_aperiodic_test
        _step("Axis x QEEG (pre-specified primary-family FDR)",
              lambda: axis_qeeg.main(config))
        _step("Axis slowing test (baseline anchor + coherence + RTM)",
              lambda: axis_slowing_test.main(config))
        _step("Axis aperiodic test (1/f-tilt mediation)",
              lambda: axis_aperiodic_test.main(config))
    else:
        print("\n[run_all] SKIP axis-QEEG suite: need qeeg_connectivity + "
              "phenotype_axis.csv (did qeeg and compass run?).")


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
        print("\n[run_all] gate: PROCEED")
        steps = _downstream_steps(config)
        print(f"[run_all] downstream steps: {sorted(steps)}")

        # --- Directional phenotyping (default): reliable-direction vMF + spherical ---
        if "directional" in steps:
            from . import directional_phenotype
            print("\n=== Directional phenotyping (vMF + spherical) ===")
            directional_phenotype.main(config)

        # --- Trajectory eval: clusterability + direction x magnitude ---
        if "trajectory" in steps:
            from . import trajectory_eval
            print("\n=== Trajectory eval (clusterability + direction x magnitude) ===")
            trajectory_eval.main(config)

        # --- Axis analysis suite (geometry -> compass -> qeeg -> axis tests) ---
        if "axis" in steps:
            print("\n" + "=" * 68 + "\n=== AXIS ANALYSIS SUITE ===\n" + "=" * 68)
            _run_axis_suite(config)

        # --- QC: line-noise / notch diagnostic ---
        if "qc" in steps:
            from . import psd_diagnostic
            _step("PSD line-noise / notch diagnostic", lambda: psd_diagnostic.main(config))

        # --- Hard clustering branch (kept but disabled by default) ---
        if "clustering" in steps:
            from . import outlier_qc, cluster, stability, qeeg, phenotype_stats, report
            print("\n=== Outlier QC (diagnostic) ===")
            outlier_qc.main(config)
            print("\n=== Module 4: cluster ===")
            cluster.main(config)
            print("\n=== Module 5: stability ===")
            stability.main(config)

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

            run_pheno = _resolve(config, "run_phenotype", _clinical_available(config))
            if run_pheno and qeeg_ok:
                print("\n=== Module 7: phenotype_stats ===")
                phenotype_stats.main(config)
            else:
                why = "no clinical table" if not run_pheno else "QEEG features not produced"
                print(f"\n[run_all] skipping module 7 (phenotype_stats): {why}.")

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
