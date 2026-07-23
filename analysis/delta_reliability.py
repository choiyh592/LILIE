"""Delta DIRECTION reliability - the decisive 'can we keep phenotyping?' test.

The between-session trajectory reverses (consecutive-step cosine ~ -0.68), which
is what you'd see if the delta DIRECTION is dominated by within-session sampling
noise rather than a real change axis. This asks that directly, from the FIXED
embeddings + trained fold models (no re-preprocessing, no re-training):

  For each progression, split its before x after segment pairs into two DISJOINT
  halves, compute the pooled delta from each half through the out-of-fold model,
  and take the cosine between the two half-deltas. High cosine => the direction is
  REPRODUCIBLE within a session (real, phenotyping defensible; the cross-session
  reversal is then genuine biological mean-reversion, not noise). Low cosine (~0)
  => the direction is sampling noise (phenotyping not recoverable from this data;
  retreat to the magnitude/control story).

Also: (a) a high-D null (cosine between DIFFERENT progressions' half-deltas), and
(b) a pair-budget sweep (reliability at 2/4/8 pairs per half) - if reliability
climbs with more pairs, averaging more segments per session would denoise the
deltas (relevant to the window / random-sampling question).

Reads progressions + deltas.npz + directional_phenotype_labels.csv + the fold
checkpoints (out/_ckpt) + embeddings. Outputs: delta_reliability.{json,png}

Run:  python -m analysis.delta_reliability --config analysis/config.yaml
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

from .config import Config, load_config, add_arg
from . import io
from . import delta as delta_mod

OI = {"blue": "#0072B2", "gray": "#999999", "vermillion": "#D55E00"}


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _load_fold_models(config: Config):
    from models.models import LILIE
    d = config["delta"]
    nf = int(config["assemble"]["num_folds"])
    models = {}
    for f in range(1, nf + 1):
        ck = os.path.join(config.out("_ckpt"), f"fold{f}", "best.ckpt")
        if os.path.exists(ck):
            models[f] = LILIE.load_from_checkpoint(
                ck, map_location="cpu",
                input_dim=int(d["input_dim"]), embedding_size=int(d["embedding_size"]),
                num_classes=2, pool_method=d["pool_method"], clf_method=d["clf_method"]).eval()
    return models


def main(config: Config) -> str:
    delta_mod._ensure_repo_on_path(config)
    d = config["delta"]
    seed = int(config["seed"]); rng = np.random.default_rng(seed)
    dpi = int(config["report"]["fig_dpi"])
    budgets = [2, 4, 8]

    prog = io.read_table(config.out("progressions"))
    prog["before_date"] = pd.to_datetime(prog["before_date"])
    prog["after_date"] = pd.to_datetime(prog["after_date"])
    seg_index = delta_mod._segment_index(config.path("paths", "metadata_csv"))
    embeddings = np.load(config.path("paths", "embeddings_npy"), mmap_mode="r")
    models = _load_fold_models(config)
    if not models:
        raise SystemExit("[reliability] no fold checkpoints under "
                         f"{config.out('_ckpt')} - run delta.py (module 2) with early_stopping.")
    oof = bool(d["out_of_fold_deltas"])

    dz = np.load(config.out("deltas.npz"), allow_pickle=True)
    magmap = dict(zip([str(x) for x in dz["progression_id"]],
                      np.linalg.norm(dz["delta"].astype(float), axis=1)))
    relmap = {}
    labp = config.out("directional_phenotype_labels.csv")
    if os.path.exists(labp):
        lab = pd.read_csv(labp)
        relmap = dict(zip(lab["progression_id"].astype(str), lab["is_reliable"].astype(bool)))

    def _half_delta(model, bi, ai, pairs):
        med, _, _ = delta_mod._progression_delta(model, embeddings, bi, ai, pairs)
        return np.asarray(med, float)

    rows, halfA = [], {}
    for _, r in prog.iterrows():
        pid, fold = int(r["patient_id"]), int(r["fold"])
        bi = seg_index.get((pid, r["before_date"].normalize()))
        ai = seg_index.get((pid, r["after_date"].normalize()))
        if not bi or not ai:
            continue
        model = models.get(fold) if oof else models.get(1)
        if model is None:
            continue
        gid = str(r["progression_id"])
        allpairs = [(b, a) for b in bi for a in ai]
        if len(allpairs) < 4:
            rows.append({"progression_id": gid, "split_half_cos": np.nan,
                         "n_pairs": len(allpairs), "magnitude": magmap.get(gid, np.nan),
                         "is_reliable": relmap.get(gid, None), "budget_cos": {}})
            continue
        perm = rng.permutation(len(allpairs)); h = len(allpairs) // 2
        da = _half_delta(model, bi, ai, [allpairs[i] for i in perm[:h]])
        db = _half_delta(model, bi, ai, [allpairs[i] for i in perm[h:2 * h]])
        bud = {}
        for bch in budgets:
            if len(allpairs) >= 2 * bch:
                pr = rng.permutation(len(allpairs))
                x1 = _half_delta(model, bi, ai, [allpairs[i] for i in pr[:bch]])
                x2 = _half_delta(model, bi, ai, [allpairs[i] for i in pr[bch:2 * bch]])
                bud[bch] = _cos(x1, x2)
        rows.append({"progression_id": gid, "split_half_cos": _cos(da, db),
                     "n_pairs": len(allpairs), "magnitude": magmap.get(gid, np.nan),
                     "is_reliable": relmap.get(gid, None), "budget_cos": bud})
        halfA[gid] = da

    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["split_half_cos"])
    rel = valid[valid["is_reliable"] == True]                          # noqa: E712
    stab = valid[valid["is_reliable"] == False]                        # noqa: E712

    # high-D null: cosine between DIFFERENT progressions' half-deltas
    gids = list(halfA); null = []
    if len(gids) >= 2:
        for _ in range(3000):
            i, j = rng.choice(len(gids), 2, replace=False)
            null.append(_cos(halfA[gids[i]], halfA[gids[j]]))
    null = np.array(null) if null else np.array([np.nan])

    def _med(s): return float(np.median(s)) if len(s) else np.nan
    med_all = _med(valid["split_half_cos"])
    med_rel = _med(rel["split_half_cos"]) if len(rel) else np.nan
    med_stab = _med(stab["split_half_cos"]) if len(stab) else np.nan
    frac_rel_reliable = float(np.mean(rel["split_half_cos"] > 0.5)) if len(rel) else np.nan
    rho_mag = (float(spearmanr(valid["magnitude"], valid["split_half_cos"])[0])
               if valid["magnitude"].notna().sum() > 3 else np.nan)
    null_med, null_p95 = float(np.median(null)), float(np.nanpercentile(null, 95))

    budget_curve = {}
    for bch in budgets:
        vals = [r["budget_cos"].get(bch) for r in rows if r["budget_cos"].get(bch) is not None]
        budget_curve[bch] = float(np.median(vals)) if vals else np.nan

    # verdict
    if np.isfinite(med_rel) and med_rel > 0.5 and med_rel > null_p95:
        verdict = ("RELIABLE: within-session delta direction reproduces (reliable-subset "
                   "median split-half cos %.2f >> null %.2f). The direction is real, so "
                   "phenotyping is defensible and the cross-session reversal is genuine "
                   "mean-reversion, not sampling noise." % (med_rel, null_p95))
    elif np.isfinite(med_rel) and med_rel > 0.3:
        verdict = ("MARGINAL: direction is partly reproducible (reliable-subset median %.2f). "
                   "Phenotyping is borderline; more segments per session (see budget curve) "
                   "or vigilance-matched sampling may be needed." % med_rel)
    else:
        verdict = ("NOISE: within-session direction does not reproduce (reliable-subset "
                   "median %.2f ~ null %.2f). The delta DIRECTION is sampling noise -> "
                   "directional phenotyping is not recoverable from these fixed predictions; "
                   "the magnitude/control story is what holds." % (med_rel, null_p95))

    # figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bins = np.linspace(-1, 1, 21)
    if len(stab):
        ax1.hist(stab["split_half_cos"], bins=bins, color=OI["gray"], alpha=0.6,
                 label=f"stable (n={len(stab)})")
    if len(rel):
        ax1.hist(rel["split_half_cos"], bins=bins, color=OI["blue"], alpha=0.7,
                 label=f"reliable (n={len(rel)})")
    ax1.axvline(null_p95, color=OI["vermillion"], ls="--", lw=1.4, label=f"null p95={null_p95:.2f}")
    ax1.axvline(med_rel, color=OI["blue"], ls="-", lw=1.6, label=f"reliable median={med_rel:.2f}")
    ax1.set_xlabel("split-half direction cosine"); ax1.set_ylabel("count")
    ax1.set_title("(a) within-session delta-direction reliability", fontsize=11)
    ax1.legend(fontsize=8, frameon=False)
    ks = sorted(budget_curve)
    ax2.plot(ks, [budget_curve[k] for k in ks], "o-", color=OI["blue"])
    ax2.axhline(null_p95, color=OI["vermillion"], ls="--", lw=1.2, label=f"null p95={null_p95:.2f}")
    ax2.set_xlabel("segment pairs per half"); ax2.set_ylabel("median split-half cosine")
    ax2.set_title("(b) reliability vs sampling budget (does more averaging help?)", fontsize=11)
    ax2.legend(fontsize=8, frameon=False)
    for a in (ax1, ax2):
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.out("delta_reliability.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    io.write_json({
        "n_tested": int(len(valid)), "n_reliable": int(len(rel)), "n_stable": int(len(stab)),
        "median_split_half_cos_all": med_all,
        "median_split_half_cos_reliable": med_rel,
        "median_split_half_cos_stable": med_stab,
        "frac_reliable_with_cos_gt_0.5": frac_rel_reliable,
        "null_median": null_med, "null_p95": null_p95,
        "spearman_cos_vs_magnitude": rho_mag,
        "budget_curve_median_cos": budget_curve,
        "verdict": verdict,
        "note": "Split-half cosine of the pooled delta from disjoint segment-pair halves "
                "(out-of-fold model, fixed embeddings). High = direction reproducible "
                "(phenotyping real); ~null = sampling noise. Budget curve rising = more "
                "segments/session would denoise.",
    }, config.out("delta_reliability.json"))

    print(f"[reliability] n={len(valid)} progressions; reliable median split-half cos="
          f"{med_rel:.2f}, stable={med_stab:.2f}, null p95={null_p95:.2f}")
    print(f"[reliability] cos-vs-magnitude Spearman r={rho_mag:.2f} "
          f"(positive => bigger changes have more reliable directions)")
    print(f"[reliability] budget curve (pairs/half -> median cos): "
          + ", ".join(f"{k}:{budget_curve[k]:.2f}" for k in ks))
    print(f"[reliability] VERDICT: {verdict}")
    return config.out("delta_reliability.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta direction reliability (split-half)")
    add_arg(parser)
    args = parser.parse_args()
    main(load_config(args.config))
