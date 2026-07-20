"""Module 8 - report: letter outputs (SCAFFOLD).

Pre-registered plan. Figure: (a) PCA scree (gate evidence, from module 3);
(b) 2-D PCA scatter by cluster; (c) stability summary; optional (d) QEEG
gradient across clusters. Table: per-cluster n, dt, baseline severity, APOE4%,
ARIA%, age, MMSE-delta, 5 primary QEEG (adjusted p, FDR).

Status: scaffolded. Panel (a) already exists as scree.png from module 3; the
remaining panels/table are produced once modules 4-7 run.
"""
from __future__ import annotations

import argparse

from .config import Config, load_config, add_arg


def main(config: Config):
    raise NotImplementedError(
        "Module 8 (report) is scaffolded. Scree panel (a) is available now as "
        "scree.png; the rest is assembled after modules 4-7.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 8 - letter outputs")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
