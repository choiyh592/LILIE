"""graded_score - the STOP route of the scree gate (SCAFFOLD).

When module 3's gate finds a rank-1 delta space, clustering is NOT run. Instead
each progression gets a scalar change score (projection on PC1 of the retained
space) and a data-driven cutpoint, per the pre-registered fallback. This keeps
the analysis honest when there is no multi-component structure to cluster.

Status: scaffolded. Consumes X_pca.npz (n_retained typically 1) from module 3.
"""
from __future__ import annotations

import argparse

from .config import Config, load_config, add_arg
from . import io


def main(config: Config):
    gate = io.read_json(config.out("gate.json"))
    if gate.get("proceed", False):
        raise SystemExit("[graded_score] gate said PROCEED; graded score not needed.")
    raise NotImplementedError(
        "graded_score is scaffolded: PC1 change score + data-driven cutpoint. "
        "Implemented if/when the gate routes here.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate STOP route - graded change score")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
