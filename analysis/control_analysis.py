"""Control analysis - are the directional phenotypes Leqembi-specific?

Projects the untreated control deltas into the SAME direction space as the
treated cohort and asks where they land:

  - Do controls mostly stay in the LOW-CHANGE region (small magnitude)? Then the
    large directional changes are treatment-associated -> STRENGTHENS the claim.
  - Do controls reach reliable magnitude AND align with a treated phenotype
    (high cosine to its mean direction)? Then the phenotype is not
    treatment-specific -> WEAKENS the claim.
  - Do controls change but in unrelated directions (low cosine to all)? The
    phenotypes' directions may still be treatment-specific.

n is tiny (a handful of control pairs), so this is a qualitative sanity check,
not a powered test - stated plainly in the output.

Reads deltas.npz + directional_phenotype_labels.csv + control_deltas.npz.
Outputs: control_analysis.json, control_analysis.png

Run:  python -m analysis.control_analysis --config analysis/config.yaml
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from scipy.stats import mannwhitneyu

from .config import Config, load_config, add_arg
from . import io
from .directional_phenotype import _unit

_PAL = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377"]


def main(config: Config) -> str:
    dp = config.get("directional_phenotype", {})
    seed = int(config["seed"])
    which = dp.get("geometry_label", "spherical_label")

    tz = np.load(config.out("deltas.npz"), allow_pickle=True)
    Tdelta = tz["delta"].astype(float)
    Tids = tz["progression_id"]
    Ut = _unit(Tdelta); Tnorm = np.linalg.norm(Tdelta, axis=1)

    lab = pd.read_csv(config.out("directional_phenotype_labels.csv")).set_index("progression_id")
    lab = lab.reindex([str(i) for i in Tids])
    labels = lab[which].to_numpy()

    cz = np.load(config.out("control_deltas.npz"), allow_pickle=True)
    Cdelta = cz["delta"].astype(float)
    Cids = cz["progression_id"]; Cpid = cz["patient_id"]
    Uc = _unit(Cdelta); Cnorm = np.linalg.norm(Cdelta, axis=1)

    # treated reliability cut + phenotype mean directions
    pct = float(dp.get("reliable_magnitude_percentile", 50))
    cut = float(np.percentile(Tnorm, pct))
    phenos = sorted(c for c in set(labels) if c >= 0)
    mu = {c: _unit(Ut[labels == c].mean(0)[None, :])[0] for c in phenos}

    # classify each control
    rows = []
    for i in range(len(Cids)):
        cosims = {c: float(Uc[i] @ mu[c]) for c in phenos}
        nearest = max(cosims, key=cosims.get) if phenos else None
        reliable = bool(Cnorm[i] >= cut)
        rows.append({"progression_id": str(Cids[i]), "patient_id": int(Cpid[i]),
                     "magnitude": float(Cnorm[i]), "reliable": reliable,
                     "assignment": (f"pheno{nearest}" if reliable else "low-change/stable"),
                     "nearest_pheno": int(nearest) if nearest is not None else None,
                     "cosine_to_nearest": float(cosims[nearest]) if phenos else None})
    cdf = pd.DataFrame(rows)

    n_reliable = int(cdf["reliable"].sum())
    frac_reliable = n_reliable / len(cdf)
    treated_reliable_rate = float(np.mean(Tnorm >= cut))
    try:
        mw = mannwhitneyu(Tnorm, Cnorm, alternative="greater")   # treated change > control?
        mw_p = float(mw.pvalue)
    except Exception:
        mw_p = None

    # verdict heuristic
    aligned = cdf[cdf["reliable"] & (cdf["cosine_to_nearest"] > 0.5)]
    if frac_reliable <= 0.34:
        verdict = ("controls mostly low-change -> phenotypes look treatment-associated "
                   "(STRENGTHENS, qualitatively)")
    elif len(aligned) >= max(1, 0.5 * n_reliable):
        verdict = ("controls reach reliable magnitude AND align with treated phenotypes "
                   "-> phenotypes may NOT be treatment-specific (WEAKENS)")
    else:
        verdict = ("controls change but in directions unlike the treated phenotypes "
                   "-> treated phenotype directions may still be specific (NEUTRAL/leans strengthen)")

    result = {
        "n_control": int(len(cdf)), "n_control_patients": int(len(set(Cpid.tolist()))),
        "treated_magnitude_cut": cut,
        "control_fraction_reliable": frac_reliable,
        "treated_fraction_reliable": treated_reliable_rate,
        "mannwhitney_treated_gt_control_p": mw_p,
        "controls": rows, "verdict": verdict,
        "caveat": "Tiny control n - qualitative sanity check, not a powered test.",
    }
    io.write_json(result, config.out("control_analysis.json"))

    # figure: shared 2D direction embedding, treated by phenotype + controls as X
    emb = PCA(n_components=2, random_state=seed).fit(Ut)
    Et, Ec = emb.transform(Ut), emb.transform(Uc)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(Et[labels < 0, 0], Et[labels < 0, 1], s=16, color="#DDDDDD", label="treated stable")
    for j, c in enumerate(phenos):
        m = labels == c
        ax.scatter(Et[m, 0], Et[m, 1], s=26, color=_PAL[j % len(_PAL)],
                   alpha=0.7, label=f"treated pheno{c}")
    for i in range(len(Cids)):
        col = "black" if not rows[i]["reliable"] else "#d62728"
        ax.scatter(Ec[i, 0], Ec[i, 1], s=130, marker="X", color=col,
                   edgecolor="white", linewidth=0.6, zorder=5)
    ax.scatter([], [], marker="X", color="#d62728", s=90, label="control (reliable)")
    ax.scatter([], [], marker="X", color="black", s=90, label="control (low-change)")
    ax.set_xlabel("dir PC1"); ax.set_ylabel("dir PC2")
    ax.set_title("Controls (X) projected into treated phenotype space")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(config.out("control_analysis.png"), dpi=int(config["report"]["fig_dpi"]))
    plt.close(fig)

    print(f"[control] {len(cdf)} controls: {n_reliable} reliable "
          f"({frac_reliable:.0%}) vs treated {treated_reliable_rate:.0%}; "
          f"MannWhitney treated>control p={mw_p}")
    print(f"[control] verdict: {verdict}")
    print(f"[control] wrote control_analysis.{{json,png}} to {config.output_dir}")
    return config.out("control_analysis.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control analysis vs treated phenotypes")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
