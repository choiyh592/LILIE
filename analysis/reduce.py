"""Module 3 - reduce: PCA on the delta matrix + go/no-go scree gate.

Z-scores each embedding dimension across progressions, fits PCA on the deltas
only, saves the scree / explained-variance curve, and evaluates the rank-1
gate. If PC1 dominates with PC2+ at the noise floor, the clustering branch is
HALTED (route -> graded_score.py); otherwise the run may proceed. The gate
decision + rationale are logged and written to gate.json (invariant 5).

Outputs (paths.output_dir):
  scree.csv, scree.png        - explained-variance evidence for the gate
  X_pca.npz                   - retained component scores + loadings
  gate.json                   - {proceed, route, rationale, ...}

Run:  python -m analysis.reduce --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from .config import Config, load_config, add_arg
from . import io
from . import invariants


def _load_deltas(config: Config):
    z = np.load(config.out("deltas.npz"), allow_pickle=True)
    return z["delta"], z["progression_id"], z["patient_id"], z["fold"]


def _plot_scree(evr, bstick, noise_floor, gate, path):
    k = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(k, evr, "o-", label="explained variance ratio", color="#3b6ea5")
    ax.plot(k, bstick, "s--", label="broken-stick null", color="#c0603a", alpha=0.8)
    ax.axhline(noise_floor, ls=":", color="#888", label=f"noise floor ({noise_floor})")
    ax.set_xlabel("principal component")
    ax.set_ylabel("proportion of variance")
    verdict = "PROCEED -> clustering" if gate["proceed"] else "STOP -> graded_score"
    ax.set_title(f"Delta-space scree | gate: {verdict}")
    ax.legend(frameon=False)
    ax.set_xticks(k[: min(len(k), 20)])
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(config: Config) -> dict:
    r = config["reduce"]
    delta, prog_ids, patient_id, fold = _load_deltas(config)

    # Invariant 3 (defensive): one row per progression before we reduce.
    invariants.assert_progression_unit(delta.shape[0], prog_ids)

    X = delta.astype(float)
    if bool(r["zscore"]):
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    max_comp = r["max_components"] or min(X.shape[0], X.shape[1])
    max_comp = int(min(max_comp, X.shape[0], X.shape[1]))
    pca = PCA(n_components=max_comp, random_state=int(config["seed"]))
    scores = pca.fit_transform(X)                      # fit on deltas only
    evr = pca.explained_variance_ratio_

    # --- gate (invariant 5) ---------------------------------------------------
    g = r["gate"]
    gate = invariants.gate_decision(
        evr,
        min_components_above_floor=int(g["min_components_above_floor"]),
        pc1_dominance_threshold=float(g["pc1_dominance_threshold"]),
        noise_floor=float(g["noise_floor"]),
        use_broken_stick=bool(g["use_broken_stick"]),
    )

    # scree table + figure
    bstick = np.array(gate["broken_stick"])
    scree_rows = np.column_stack([np.arange(1, len(evr) + 1), evr,
                                  np.cumsum(evr), bstick])
    scree_path = config.out("scree.csv")
    np.savetxt(scree_path, scree_rows, delimiter=",",
               header="component,explained_variance_ratio,cumulative,broken_stick",
               comments="")
    _plot_scree(evr, bstick, float(g["noise_floor"]), gate, config.out("scree.png"))

    # retained components: those above floor (>=1 always kept for the score branch)
    signal = np.array(gate["signal_mask"])
    n_keep = max(int(signal.sum()), 1)
    np.savez(config.out("X_pca.npz"),
             X_pca=scores[:, :n_keep],
             components=pca.components_[:n_keep],
             explained_variance_ratio=evr,
             progression_id=prog_ids, patient_id=patient_id, fold=fold,
             n_retained=n_keep)

    io.write_json(gate, config.out("gate.json"))

    print("[reduce] explained variance ratio:",
          np.array2string(evr[: min(len(evr), 8)], precision=3))
    print(f"[reduce] GATE: {gate['route'].upper()} -- {gate['rationale']}")
    print(f"[reduce] scree.png + gate.json written to {config.output_dir}")
    return gate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3 - PCA + scree gate")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
