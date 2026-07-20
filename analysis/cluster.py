"""Module 4 - cluster: clustering + k selection (SCAFFOLD).

Pre-registered plan: k-means (primary) and GMM (sensitivity) on the retained
PCs from module 3; select k by silhouette + gap statistic, favoring small k.
k is NOT chosen by downstream phenotype separation.

Status: scaffolded. Enabled only after the scree gate returns PROCEED and the
explained-variance curve has been inspected. Input contract: X_pca.npz from
module 3; output: per-progression cluster labels (labels.npz) keyed by
progression_id (invariant 3 - the unit stays the progression).
"""
from __future__ import annotations

import argparse

from .config import Config, load_config, add_arg
from . import io


def main(config: Config):
    gate = io.read_json(config.out("gate.json"))
    if not gate.get("proceed", False):
        raise SystemExit("[cluster] gate routed to graded_score; clustering is skipped.")
    raise NotImplementedError(
        "Module 4 (cluster) is scaffolded. It is implemented after the scree "
        "gate is inspected and returns PROCEED. See README 'Roadmap'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4 - clustering + k selection")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
