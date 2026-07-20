"""Module 5 - stability: bootstrap cluster stability (SCAFFOLD).

Pre-registered plan (load-bearing at this n): clusterwise Jaccard (Hennig) per
cluster, flag < 0.60; ARI / NMI / Cohen's kappa across bootstrap resamples;
sensitivity over pooler variant, k +/- 1, and k-means vs GMM. Bootstrap
resamples MUST respect patient groups (invariant 1) -
``invariants.assert_resample_groups`` guards this.

Status: scaffolded; enabled with module 4 after the gate.
"""
from __future__ import annotations

import argparse

from .config import Config, load_config, add_arg
from . import io


def main(config: Config):
    gate = io.read_json(config.out("gate.json"))
    if not gate.get("proceed", False):
        raise SystemExit("[stability] gate routed to graded_score; stability is skipped.")
    raise NotImplementedError(
        "Module 5 (stability) is scaffolded. Bootstrap resamples will use "
        "invariants.assert_resample_groups to keep patients disjoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 5 - bootstrap stability")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
