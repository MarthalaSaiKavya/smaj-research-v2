#!/usr/bin/env python
"""Build the paper figures from the raw v2 result files (and v1 numbers for the comparison overlays).

Usage (repo root):  python analysis/make_paper_figures.py      (run derive_stats.py first)
Writes paper/figures/fig_protocol, fig_layers, fig_goal2, fig_heldout (.pdf + .png) and copies run figures used in the appendix.
Colour follows the entity in every figure: blue = selected occlusion features, orange = colour control, grey = random controls,
aqua = whole-MLP quantities (all transcoder features, top-K by magnitude, MLP swap), ink = full residual swap / frames.
Categorical slots 1-3 of the reference palette (validated all-pairs: CVD dE >= 9.2, normal >= 24); grey is a neutral, not a slot.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((ROOT / "analysis" / "derived_stats.json").read_text())
V1 = json.loads((ROOT / "analysis" / "v1_derived_stats.json").read_text())

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, GRID = "#8c8b86", "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: conditions (crop rows from the run's own render)
def protocol():
    im = Image.open(R / "fig_probe_conditions_agentview.png").convert("RGB")
    rows = [(81 + 202 * i, 261 + 202 * i) for i in range(10)]
    cols = [(45, 224), (238, 418), (432, 611), (625, 805), (819, 999), (1013, 1192), (1206, 1386)]
    names = ["base", "recolor", "absent", "occluded", "slab_miss", "occluded_absent", "occluded_distractor"]
    pick = [0, 1]  # task 0 (demo) and task 1 (rollout), discovery frames at phase 0.8
    fig, axes = plt.subplots(len(pick), 7, figsize=(6.8, 2.2), gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    for r, ri in enumerate(pick):
        for c, (x0, x1) in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(np.asarray(im.crop((x0 + 1, rows[ri][0] + 1, x1, rows[ri][1]))))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(names[c].replace("_", "\n", 1) if "_" in names[c] else names[c], fontsize=7, loc="center", fontweight="normal")
        axes[r][0].set_ylabel(["task 0, $t{=}39$", "task 1, $t{=}51$"][r], fontsize=7)
    save(fig, "fig_protocol")


# ---------------------------------------------------------------- Fig 2: layer profile, v1 vs v2
def layers():
    c4 = S["check4"]
    L = np.arange(18)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.95), gridspec_kw={"width_ratios": [1.35, 1, 1], "wspace": 0.42})
    ax = axes[0]
    style(ax)
    ax.plot(L, c4["resid_swap_mean"], color=INK, lw=1.8, marker="o", ms=3.2, label="residual swap")
    ax.plot(L, V1["check4"]["resid_swap_mean"], color=INK, lw=0.9, ls="--")
    ax.plot(L, c4["mlp_swap_mean"], color=AQUA, lw=1.8, marker="s", ms=3.2, label="MLP swap")
    ax.plot(L, V1["check4"]["mlp_swap_mean"], color=AQUA, lw=0.9, ls="--")
    ax.plot([], [], color=INK2, lw=0.9, ls="--", label="v1 (4 frames)")
    ax.set_ylim(-3, 104)
    ax.set_xticks(range(0, 18, 3))
    ax.set_xlabel("layer swapped (base$\\rightarrow$occluded)")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("(a) where the signal lives")
    ax.legend(fontsize=6.3, loc="upper right", bbox_to_anchor=(1.02, 1.02), handlelength=1.6, labelspacing=0.2)
    ax = axes[1]
    style(ax)
    n2 = [S["goal1"]["n_pass_per_layer"][str(l)] for l in L]
    n1 = [V1["goal1"]["n_pass_per_layer"][str(l)] for l in L]
    ax.bar(L, n2, width=0.72, color=BLUE, label=f"v2: {sum(n2)} of 36,864")
    ax.step(np.r_[L - 0.5, 17.5], np.r_[n1, n1[-1]], where="post", color=INK2, lw=0.8, label=f"v1: {sum(n1)} of 9,216")
    ax.set_xticks(range(0, 18, 3))
    ax.set_xlabel("layer")
    ax.set_ylabel("# selective features")
    ax.set_title("(b) Goal 1 features")
    ax.set_ylim(0, 12.5)
    ax.legend(fontsize=6.3, loc="upper right", handlelength=1.4, labelspacing=0.25)
    ax = axes[2]
    style(ax)
    share = [S["edit_mass"]["share_by_layer_pct"][str(l)] for l in L]
    ax.bar(L, share, width=0.72, color=AQUA)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 150)
    ax.set_xticks(range(0, 18, 3))
    ax.set_xlabel("layer")
    ax.set_ylabel("share of activation change (%)")
    ax.set_title("(c) where features change")
    ax.text(16.9, 95, f"L16–17: {S['edit_mass']['share_layers_16_17_pct']:.0f}%", ha="right", va="bottom", fontsize=6.5, color=INK)
    save(fig, "fig_layers")


# ---------------------------------------------------------------- Fig 3: Goal 2 on held-out frames + k-curve
def goal2():
    import csv

    with open(R / "goal2_rows.csv") as f:
        rows = list(csv.DictReader(f))

    def per_task(method):
        by = {}
        for r in rows:
            if r["layer"] == "-2" and r["method"] == method and r["scope"] == "all" and r["direction"] == "inject":
                by.setdefault(int(r["task_id"]), {}).setdefault(int(r["frame"]), []).append(float(r["closed_pct"]))
        return {t: np.mean([np.mean(v) for v in fr.values()]) for t, fr in sorted(by.items())}

    G = S["goal2"]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.45), gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.3})
    ax = axes[0]
    style(ax)
    meths = [("occ_features", "occlusion\n(29)", BLUE), ("random_same_norm", "random\nsame-norm", GREY),
             ("random_raw", "random\nraw", GREY), ("color_features", "colour\n(34)", ORANGE)]
    rng = np.random.default_rng(0)
    for i, (m, lab, col) in enumerate(meths):
        pt = per_task(m)
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(pt)), list(pt.values()), s=7, color=col, alpha=0.55, lw=0, zorder=2)
        lo, hi = G["ci95"][m]
        ax.errorbar([i], [G["means"][m]], yerr=[[G["means"][m] - lo], [hi - G["means"][m]]], fmt="o", ms=4.5, color=col,
                    mec="white", mew=0.6, capsize=2.5, lw=1.2, zorder=3)
    thr = G["means"]["random_same_norm"] + G["margin_required"]
    ax.axhline(thr, color=INK2, lw=0.7, ls="--")
    ax.text(3.45, thr + 0.35, "pre-registered pass line (random + 10 pts)", ha="right", va="bottom", fontsize=6.3, color=INK2)
    ax.plot([0], [S["v1"]["means"]["occ_features"]], marker="D", ms=4.5, mfc="white", mec=BLUE, mew=1.0, ls="none", zorder=4)
    ax.annotate("v1 (4 selected frames)", (0, S["v1"]["means"]["occ_features"]), xytext=(0.28, 5.6), fontsize=6.3, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(len(meths)))
    ax.set_xticklabels([m[1] for m in meths], fontsize=7)
    ax.set_ylim(-2.2, 11.8)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("gap closed (%), all 18 layers")
    ax.set_title("(a) selected features vs. controls")
    ax = axes[1]
    style(ax)
    style(ax, "x")
    kc = G["kcurve"]
    for fam, ls, lab, col in (("oracle", "-", "top-$K$ by this frame's activation change", AQUA),
                              ("discovery", "--", "top-$K$ by discovery-frame change", AQUA), ("random", "-", "random $K$", GREY)):
        ks = sorted(kc[fam], key=int)
        x = [int(k) for k in ks]
        y = [kc[fam][k]["mean"] for k in ks]
        ax.fill_between(x, [kc[fam][k]["ci95"][0] for k in ks], [kc[fam][k]["ci95"][1] for k in ks], color=col, alpha=0.13, lw=0)
        ax.plot(x, y, ls=ls, color=col, lw=1.5, marker="o", ms=3, label=lab)
    ax.plot([29], [G["means"]["occ_features"]], marker="*", ms=9, color=BLUE, mec="white", mew=0.5, ls="none", label="29 selected occlusion features")
    for val, ls, lab in ((G["means"]["all_tc_features"], "--", "all 36,864 features"), (G["means"]["mlp_swap_ceiling"], ":", "MLP-output swap (ceiling)")):
        ax.axhline(val, color=INK2, lw=0.8, ls=ls)
        ax.text(38000, val - 1.4, f"{lab}: {val:.1f}%", fontsize=6.3, color=INK2, va="top", ha="right")
    ax.set_xscale("log")
    ax.set_xlim(20, 40000)
    ax.set_ylim(-3, 84)
    ax.set_xlabel("features patched (all 18 layers)")
    ax.set_ylabel("gap closed (%)")
    ax.set_title("(b) how many features does it take?")
    ax.legend(fontsize=6.2, loc="upper left", bbox_to_anchor=(0.0, 0.87), handlelength=2.0, labelspacing=0.2)
    save(fig, "fig_goal2")


# ---------------------------------------------------------------- Fig 4: held-out replication and target specificity
def heldout():
    import csv

    rk = torch.load(R / "goal1_rank.pt", weights_only=False)
    X = np.load(R / "goal2_test_feature_sums.npy")
    Ls = sorted(rk["edit"])
    typ = np.stack([rk["typical"][l].numpy() for l in Ls])
    alive = np.stack([rk["alive"][l].numpy() for l in Ls]).astype(bool)
    disc = np.stack([rk["d_occluded"][l].numpy() for l in Ls]) / np.maximum(typ, 1e-8)
    test = (X[:, 3] - X[:, 0]).mean(0) / np.maximum(typ, 1e-8)
    sel = json.loads((R / "goal1_selected.json").read_text())
    selS = [(int(l), j) for l, v in sel["layers"].items() for j in v["occ"]]
    repl = set(S["heldout_recomputed"]["replicating_features"])
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    style(ax, "both")
    ax.scatter(disc[alive], test[alive], s=1.2, color=GREY, alpha=0.25, lw=0, rasterized=True, label=f"all {alive.sum():,} live features")
    for (l, j) in selS:
        ok = f"L{l}f{j}" in repl
        ax.scatter([disc[l, j]], [test[l, j]], s=16, marker="o", facecolor=BLUE if ok else "white", edgecolor=BLUE, lw=0.9, zorder=3)
    ax.scatter([], [], s=16, facecolor=BLUE, edgecolor=BLUE, label=f"selected, replicates on test ({len(repl)})")
    ax.scatter([], [], s=16, facecolor="white", edgecolor=BLUE, label=f"selected, does not ({len(selS) - len(repl)})")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    lim = [-30, 150]
    ax.plot(lim, lim, color=INK2, lw=0.5, ls=":")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("occlusion response, discovery frames\n(token-firings above base)")
    ax.set_ylabel("occlusion response, test frames")
    h = S["goal2"]["heldout"]
    ax.set_title(f"(a) held-out replication ($\\rho$ = {h['spearman_d_occluded_discovery_vs_test_all_alive']:.2f})")
    ax.legend(fontsize=6.1, loc="upper left", handlelength=1.0, labelspacing=0.25, markerscale=1.0)
    ax = axes[1]
    style(ax, "both")
    rows = json.loads((R / "validate.json").read_text())["check3"]["rows"]
    tg = np.array([r["gap_occluded"] for r in rows])
    ds = np.array([r["rmse_occluded_distractor"] for r in rows])
    sp = np.array([r["split"] == "test" for r in rows])
    ax.scatter(ds[~sp], tg[~sp], s=10, color=INK, marker="^", lw=0, alpha=0.8, label="discovery frames")
    ax.scatter(ds[sp], tg[sp], s=10, color=INK, marker="o", lw=0, alpha=0.8, label="test frames")
    ax.plot([0.01, 3], [0.01, 3], color=INK2, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 3)
    ax.set_ylim(0.01, 3)
    ax.set_xlabel("action change: paint the distractor bowl")
    ax.set_ylabel("action change: paint the target bowl")
    td = S["check3"]["target_vs_distractor"]
    ax.set_title(f"(b) the action tracks the target ({td['n_target_gt_distractor']}/{td['n']} above diagonal)")
    ax.legend(fontsize=6.3, loc="lower right", handlelength=1.0)
    save(fig, "fig_heldout")


# ---------------------------------------------------------------- appendix: closed-loop frames and run figures
def appendix():
    import cv2

    tiles = []
    for name in ("clean_reference", "baseline_hidden"):
        cap = cv2.VideoCapture(str(R / "videos" / f"{name}_ep0.mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        row = []
        for k in (0, n // 3, 2 * n // 3, n - 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, fr = cap.read()
            row.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            row.append(np.full((fr.shape[0], 4, 3), 255, np.uint8))
        tiles.append(np.concatenate(row[:-1], 1))
        tiles.append(np.full((4, tiles[-1].shape[1], 3), 255, np.uint8))
    Image.fromarray(np.concatenate(tiles[:-1], 0)).save(FIG / "fig_closed_loop.png")
    for f in ("fig_goal1_contact_sheet.png", "fig_goal1_features.png", "fig_check5_transcoders.png"):
        shutil.copy2(R / f, FIG / f)
    for f in ("fig_probe_conditions_agentview.png", "fig_probe_conditions_wrist.png"):
        im = Image.open(R / f).convert("RGB")
        im.thumbnail((1100, 1650))
        im.save(FIG / f, optimize=True)


if __name__ == "__main__":
    protocol()
    layers()
    goal2()
    heldout()
    appendix()
    print("figures:", sorted(p.name for p in FIG.iterdir()))
