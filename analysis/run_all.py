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
import sys

from .config import Config, load_config, add_arg
from . import invariants
from . import assemble, delta, reduce as reduce_mod


STEPS_PRE_GATE = ["assemble", "delta", "reduce"]


def _ensure_repo_on_path(config: Config):
    repo_root = config.path("repo_root")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


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

    # Gate routing (modules 4-8 / graded_score are scaffolded).
    if invariants.should_proceed(gate):
        print("\n[run_all] gate: PROCEED -> clustering branch")
        from . import cluster, stability, qeeg, phenotype_stats, report
        for name, mod in [("cluster", cluster), ("stability", stability),
                          ("qeeg", qeeg), ("phenotype_stats", phenotype_stats),
                          ("report", report)]:
            print(f"\n=== {name} ===")
            mod.main(config)
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
    args = parser.parse_args()
    run(load_config(args.config), start_from=args.start_from)
