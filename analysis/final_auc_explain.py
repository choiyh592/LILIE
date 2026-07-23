"""Honest ordering-AUROC figures + an example CAM explainability plot.

Two things, kept separate and un-spun:

  final_auc_perfold.png  Per-fold-model AUROC. If the raw held-out predictions were
                         saved (ordering_predictions.npz, written by delta.py), each
                         fold's ROC curve is drawn plus the mean±std of the per-fold
                         AUCs; otherwise the per-fold AUC values are shown as points
                         with the mean±std band. Small per-fold n is stated, not hidden.
  final_auc_oof.png      The pooled OUT-OF-FOLD AUROC (the honest number) with its
                         patient-clustered 95% CI, the rank-calibrated pooled AUROC,
                         the per-fold mean for reference, the permutation p, and — in
                         full view — the early-stopping-monitors-the-test-fold optimism
                         caveat. Nothing is dropped to make the number look better.
  final_cam_example.png  An example saliency/CAM map over raw EEG for the stitched
                         LaBraM+LILIE model, using the repo's own gradient computation
                         (explain/.../saliency_map_LaBraM.py). With --real plus the
                         checkpoints/HDF5 it renders a genuine sample; otherwise it
                         renders a clearly-labelled ILLUSTRATIVE synthetic example so
                         the output format is documented without a GPU.

Reads: ordering_auc.json (always), ordering_predictions.npz (optional, for ROC curves).
Run:   python -m analysis.final_auc_explain --config analysis/config.yaml
       python -m analysis.final_auc_explain --config analysis/config.yaml --real \
           --hdf5 PATH --labram_ckpt PATH --lilie_ckpt PATH --group0 G0 --group1 G1
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, load_config, add_arg
from . import io

OI = {"blue": "#0072B2", "vermillion": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "gray": "#999999", "black": "#000000", "purple": "#CC79A7"}


def _jload(config, name):
    p = config.out(name)
    try:
        return io.read_json(p) if os.path.exists(p) else None
    except Exception:
        return None


def _preds(config):
    p = config.out("ordering_predictions.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def _roc(prob, label):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(label, prob)
    return fpr, tpr, float(roc_auc_score(label, prob))


# --------------------------------------------------------------------------- #
# figure: per-fold AUROC
# --------------------------------------------------------------------------- #
def fig_perfold(config, aj, preds):
    dpi = int(config["report"]["fig_dpi"])
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    pf = aj.get("perfold_auc", {}) if aj else {}
    pf_mean = aj.get("perfold_auc_mean") if aj else None
    pf_std = aj.get("perfold_auc_std") if aj else None

    if preds is not None and "fold" in preds:
        folds = sorted(set(int(f) for f in preds["fold"]))
        cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(folds)))
        aucs = []
        for c, f in zip(cmap, folds):
            m = preds["fold"].astype(int) == f
            if len(set(preds["label"][m].tolist())) < 2:
                continue
            fpr, tpr, a = _roc(preds["prob"][m], preds["label"][m])
            aucs.append(a)
            ax.plot(fpr, tpr, color=c, lw=1.6, alpha=0.9,
                    label=f"fold {f}: AUC={a:.2f} (n={int(m.sum())})")
        ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="#999", lw=1)
        mu = float(np.mean(aucs)) if aucs else float("nan")
        sd = float(np.std(aucs)) if aucs else float("nan")
        ax.text(0.98, 0.05, f"per-fold AUC = {mu:.2f} ± {sd:.2f}  ({len(aucs)} folds)",
                transform=ax.transAxes, ha="right", fontsize=9,
                bbox=dict(boxstyle="round", fc="#f4f7fb", ec="#cfd8e3"))
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.legend(fontsize=7.5, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.16))
    else:
        # summary fallback: per-fold AUC points + mean±std band
        items = sorted(((int(k), float(v)) for k, v in pf.items())) if pf else []
        if items:
            xs = [k for k, _ in items]; ys = [v for _, v in items]
            if pf_mean is not None and pf_std is not None:
                ax.axhspan(pf_mean - pf_std, pf_mean + pf_std, color=OI["blue"], alpha=0.12,
                           label=f"mean ± std = {pf_mean:.2f} ± {pf_std:.2f}")
                ax.axhline(pf_mean, color=OI["blue"], lw=1.6)
            ax.scatter(xs, ys, s=90, color=OI["blue"], edgecolor="white", zorder=4)
            for k, v in items:
                ax.text(k, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
            ax.axhline(0.5, ls=(0, (4, 4)), color="#999", lw=1, label="chance")
            ax.set_xticks(xs); ax.set_xticklabels([f"fold {k}" for k in xs], fontsize=9)
            ax.set_ylim(0.0, 1.0); ax.set_ylabel("held-out AUROC")
            ax.legend(fontsize=8.5, frameon=False, loc="upper left")
            ax.text(0.02, 0.03, "per-fold n is small → these are noisy;\nsee OOF figure for the pooled estimate",
                    transform=ax.transAxes, fontsize=8, color="#555")
        else:
            ax.text(0.5, 0.5, "ordering_auc.json missing", ha="center", transform=ax.transAxes)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = config.out("final_auc_perfold.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# figure: out-of-fold AUROC (the honest number)
# --------------------------------------------------------------------------- #
def fig_oof(config, aj, preds):
    dpi = int(config["report"]["fig_dpi"])
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    auc = aj.get("ordering_auc") if aj else None
    ci = aj.get("ordering_auc_ci95") if aj else None
    cauc = aj.get("calibrated_pooled_auc") if aj else None
    cci = aj.get("calibrated_pooled_auc_ci95") if aj else None
    pfm = aj.get("perfold_auc_mean") if aj else None
    pfs = aj.get("perfold_auc_std") if aj else None
    perm = aj.get("permutation_pvalue") if aj else None
    es = aj.get("early_stopping_monitors_test_fold") if aj else None
    npat = aj.get("n_patients") if aj else None
    npred = aj.get("n_predictions") if aj else None

    if preds is not None and len(set(preds["label"].tolist())) >= 2:
        fpr, tpr, a = _roc(preds["prob"], preds["label"])
        ax.plot(fpr, tpr, color=OI["blue"], lw=2.6,
                label=f"pooled OOF: AUC={a:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else ""))
        if "prob_calibrated" in preds:
            fprc, tprc, ac = _roc(preds["prob_calibrated"], preds["label"])
            ax.plot(fprc, tprc, color=OI["green"], lw=1.8, ls=(0, (3, 2)),
                    label=f"rank-calibrated OOF: AUC={ac:.2f}" + (f" [{cci[0]:.2f}, {cci[1]:.2f}]" if cci else ""))
        ax.plot([0, 1], [0, 1], ls=(0, (4, 4)), color="#999", lw=1, label="chance (0.5)")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    else:
        # summary fallback: three point estimates with error bars
        rows = []
        if auc is not None:
            rows.append(("pooled OOF", auc, ci, OI["blue"]))
        if cauc is not None:
            rows.append(("rank-calibrated OOF", cauc, cci, OI["green"]))
        if pfm is not None:
            rows.append(("per-fold mean", pfm, [pfm - (pfs or 0), pfm + (pfs or 0)], OI["gray"]))
        y = np.arange(len(rows))[::-1]
        for yi, (name, val, cin, col) in zip(y, rows):
            lo, hi = (cin if cin else [val, val])
            ax.plot([lo, hi], [yi, yi], color=col, lw=3, solid_capstyle="round", zorder=3)
            ax.scatter([val], [yi], s=90, color=col, edgecolor="white", zorder=4)
            ax.text(val, yi + 0.12, f"{val:.2f}", ha="center", fontsize=9)
        ax.axvline(0.5, ls=(0, (4, 4)), color="#999", lw=1)
        ax.text(0.5, -0.6, "chance", ha="center", fontsize=8, color="#777")
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_xlim(0.4, 1.0); ax.set_xlabel("AUROC")
        ax.set_ylim(-1.0, len(rows) - 0.4)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

    caveat = []
    if perm is not None:
        caveat.append(f"permutation p = {perm:.3f}")
    if npat is not None and npred is not None:
        caveat.append(f"{npred} held-out predictions / {npat} patients")
    if es:
        caveat.append("CAVEAT: early stopping monitored the held-out fold\n"
                      "(val=test) → all OOF AUCs are optimistically biased")
    if caveat:
        ax.text(0.02, 0.98, "\n".join(caveat), transform=ax.transAxes, va="top", fontsize=8.2,
                bbox=dict(boxstyle="round", fc="#fdf1ec" if es else "#f4f7fb",
                          ec="#e3b9a6" if es else "#cfd8e3"))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = config.out("final_auc_oof.png"); fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# figure: example CAM / saliency (real via repo, else illustrative synthetic)
# --------------------------------------------------------------------------- #
_CH = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8",
       "T3", "T4", "T5", "T6", "A1", "A2", "Fz", "Cz", "Pz", "T1", "T2"]


def _plot_cam(eeg, cam, ch_names, title, save_path, dpi):
    """Repo-style overlay: each channel's signal with its gradient-magnitude CAM as a
    background strip. Kept close to explain/.../plot_eeg_with_cam but tidied."""
    fig, ax = plt.subplots(figsize=(13, 8))
    n_ch, n_t = eeg.shape
    t = np.arange(n_t)
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    offset, yt = 0, []
    for i in range(n_ch):
        s = (eeg[i] - eeg[i].mean()) / (eeg[i].std() + 1e-8)
        ax.plot(t, s + offset, color="black", lw=0.7, zorder=3)
        ax.imshow(cam_norm[i].reshape(1, -1), cmap="jet", aspect="auto",
                  extent=[0, n_t, offset - 2.5, offset + 2.5], alpha=0.5, zorder=1)
        yt.append(offset); offset += 6
    ax.set_yticks(yt); ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("timepoints"); ax.set_ylim(-4, offset)
    ax.set_title(title, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cb = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02); cb.set_label("importance (|∂ logit / ∂ EEG|)", fontsize=8)
    fig.tight_layout(); fig.savefig(save_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def _synth_cam(seed=0, n_ch=23, n_t=2000, fs=200):
    """An ILLUSTRATIVE example: alpha/theta-ish EEG + a saliency map concentrated in
    posterior channels and a couple of time windows (documents the output format;
    NOT real model output)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_t) / fs
    eeg = np.zeros((n_ch, n_t))
    for i in range(n_ch):
        alpha = (0.6 + 0.5 * (_CH[i] in ("O1", "O2", "P3", "P4", "Pz"))) * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 6))
        theta = 0.4 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 6))
        eeg[i] = alpha + theta + 0.5 * rng.standard_normal(n_t)
    # saliency: high on posterior channels within two windows, smoothed
    cam = 0.05 * rng.random((n_ch, n_t))
    post = [i for i, c in enumerate(_CH) if c in ("O1", "O2", "P3", "P4", "Pz", "T5", "T6")]
    for w0 in (int(0.25 * n_t), int(0.7 * n_t)):
        win = np.exp(-0.5 * ((np.arange(n_t) - w0) / (0.05 * n_t)) ** 2)
        for i in post:
            cam[i] += (0.8 + 0.4 * rng.random()) * win
    from scipy.ndimage import gaussian_filter1d
    cam = gaussian_filter1d(cam, sigma=10, axis=1)
    return eeg, cam


def fig_cam(config, args):
    dpi = int(config["report"]["fig_dpi"])
    out = config.out("final_cam_example.png")
    if getattr(args, "real", False):
        try:
            return _real_cam(config, args, out, dpi)
        except Exception as e:  # noqa: BLE001
            print(f"[final_auc_explain] --real CAM failed ({type(e).__name__}: {e}); "
                  "falling back to the illustrative example.")
    eeg, cam = _synth_cam()
    _plot_cam(eeg, cam, _CH,
              "Example saliency / CAM (ILLUSTRATIVE, synthetic) — run --real with checkpoints for a data sample",
              out, dpi)
    return out


def _real_cam(config, args, out, dpi):
    """Genuine map via the repo's stitched model. Imports torch/LaBraM lazily; expects
    --hdf5/--labram_ckpt/--lilie_ckpt/--group0/--group1 (see saliency_map_LaBraM.py)."""
    import sys
    import torch
    from scipy.ndimage import gaussian_filter1d
    repo = config.path("repo_root")
    for p in (repo, os.path.dirname(repo)):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from timm.models import create_model
    import LaBraM.utils as utils
    from LILIE.explain.LaBraM_Goes_Here.saliency_map_LaBraM import (
        EndToEndLongitudinal, load_raw_eeg_snippet, load_labram_checkpoint, compute_eeg_saliency)
    from LILIE.models.models import LILIE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labram = create_model(args.model, pretrained=False, num_classes=0, drop_rate=0,
                          drop_path_rate=0.1, attn_drop_rate=0.0, drop_block_rate=None,
                          use_mean_pooling=True, init_scale=0.001, use_rel_pos_bias=False,
                          use_abs_pos_emb=True, init_values=0.1, qkv_bias=False, EEG_size=3200)
    labram = load_labram_checkpoint(labram, args.labram_ckpt)
    lilie = LILIE.load_from_checkpoint(args.lilie_ckpt, input_dim=args.labram_embed_dim,
                                       embedding_size=128, num_classes=2,
                                       pool_method="Attentive", clf_method="NN")
    model = EndToEndLongitudinal(labram, lilie, segment_size=args.patch_size).to(device)
    eeg0 = load_raw_eeg_snippet(args.hdf5, args.group0, args.start0, window_size=args.window_size).to(device)
    eeg1 = load_raw_eeg_snippet(args.hdf5, args.group1, args.start1, window_size=args.window_size).to(device)
    ch_caps = [c.upper() for c in _CH]
    input_chans = utils.get_input_chans(ch_caps)
    if args.target_class == 1:
        logits, cam1, cam0 = compute_eeg_saliency(model, eeg1, eeg0, input_chans, 1)
    else:
        logits, cam0, cam1 = compute_eeg_saliency(model, eeg0, eeg1, input_chans, 0)
    prob = torch.softmax(torch.tensor(logits), dim=1)[0].numpy()
    _plot_cam(eeg0.cpu().detach().numpy()[0], cam0, _CH,
              f"Saliency / CAM — {args.group0} (target class {args.target_class}; "
              f"pred [{prob[0]:.2f}, {prob[1]:.2f}])", out, dpi)
    print(f"[final_auc_explain] real CAM: pred prob = [{prob[0]:.4f}, {prob[1]:.4f}]")
    return out


def main(config: Config, args=None) -> str:
    aj = _jload(config, "ordering_auc.json")
    preds = _preds(config)
    outs = [fig_perfold(config, aj, preds), fig_oof(config, aj, preds), fig_cam(config, args or argparse.Namespace())]
    for o in outs:
        print(f"[final_auc_explain] wrote {os.path.basename(o)}")
    if preds is None:
        print("[final_auc_explain] NOTE: ordering_predictions.npz absent → AUROC shown from the "
              "summary (points + mean±std). Re-run delta.py to persist predictions for full ROC curves.")
    return outs[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honest AUROC figures + example CAM")
    add_arg(parser)
    parser.add_argument("--real", action="store_true", help="render a genuine CAM via the repo model")
    parser.add_argument("--hdf5", type=str, default=None)
    parser.add_argument("--labram_ckpt", type=str, default=None)
    parser.add_argument("--lilie_ckpt", type=str, default=None)
    parser.add_argument("--group0", dest="group0", type=str, default=None)
    parser.add_argument("--group1", dest="group1", type=str, default=None)
    parser.add_argument("--start0", type=int, default=0)
    parser.add_argument("--start1", type=int, default=0)
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--model", type=str, default="labram_base_patch200_200")
    parser.add_argument("--window_size", type=int, default=3200)
    parser.add_argument("--patch_size", type=int, default=200)
    parser.add_argument("--labram_embed_dim", type=int, default=768)
    args = parser.parse_args()
    main(load_config(args.config), args)
