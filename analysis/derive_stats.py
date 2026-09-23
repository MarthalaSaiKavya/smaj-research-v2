#!/usr/bin/env python
"""Recompute every number quoted in REPORT.md and paper/main.tex from the raw v2 result files.

Usage (repo root):  python analysis/derive_stats.py   ->  analysis/derived_stats.json
Needs numpy and torch (CPU is enough). v1 numbers come from analysis/v1_derived_stats.json, which is a verbatim copy
of analysis/derived_stats.json in github.com/MarthalaSaiKavya/smaj-research (commit 5fc0c90).
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
J = lambda p: json.loads((R / p).read_text())
CONDS = ["base", "recolor", "absent", "occluded", "slab_miss", "occluded_absent", "occluded_distractor"]


def f(x):
    return float(x) if x is not None else None


def binom_two_sided(k, n):
    """Exact two-sided sign-test p-value."""
    from math import comb

    p = sum(comb(n, i) for i in range(n + 1) if comb(n, i) <= comb(n, k)) / 2 ** n
    return min(1.0, p)


def cluster_boot(v, c, n_boot=10000, seed=0):
    v, c = np.asarray(v, float), np.asarray(c)
    uniq = sorted(set(c.tolist()))
    S = np.array([v[c == u].sum() for u in uniq])
    N = np.array([(c == u).sum() for u in uniq], float)
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    b = S[idx].sum(1) / N[idx].sum(1)
    return float(v.mean()), [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


S = {}
cfg = J("config.json")
st = {k: J(f"status/{k}.json") for k in ("frames", "validate", "capture", "train", "goal1", "goal2", "goal3", "timings")}
log = (R / "run_v2.log").read_text()

# ------------------------------------------------------------------ setup and frame sources
no_demo = sorted(int(t) for t in re.findall(r"task (\d+): no demo file", log))
rollouts = [(int(a), int(b), c, int(d), e == "True") for a, b, c, d, e in
            re.findall(r"task (\d+): rollout init_state=(\d+) \((\w+)\): (\d+) steps, success=(\w+)", log)]
probe = st["frames"]["probe"]
failed = {(t, i) for t, i, _, _, ok in rollouts if not ok}
probe_from_failed = [p["frame_id"] for p in probe
                     if "rollout" in p["episode"] and (p["task_id"], int(p["episode"].split("init")[-1])) in failed]
ag = np.array([p["edit_info"]["image"]["mask_px"] for p in probe])
wr = np.array([p["edit_info"]["image2"]["mask_px"] for p in probe])
dag = np.array([p["edit_info"]["image"]["distractor_px"] for p in probe])
H = W = int(cfg["obs_size"])
S["setup"] = {
    "policy": cfg["policy_path"], "suite": cfg["suite"], "task_ids": cfg["task_ids"], "n_tasks": len(cfg["task_ids"]),
    "n_probe_frames": st["frames"]["n_probe_frames"], "n_discovery_frames": st["frames"]["n_discovery_frames"],
    "n_test_frames": st["frames"]["n_test_frames"], "n_extra_train_frames": st["frames"]["n_extra_train_frames"],
    "n_samples": st["capture"]["n_samples"], "n_tokens": st["capture"]["n_tokens"], "n_layers": st["capture"]["n_layers"],
    "tc_features": cfg["tc_features"], "tc_k": cfg["tc_k"], "tc_steps": cfg["tc_steps"], "n_features_total": cfg["tc_features"] * 18,
    "frame_phases": cfg["frame_phases"], "discovery_demos": cfg["discovery_demos"], "test_demos": cfg["test_demos"],
    "extra_demos": cfg["extra_demos"], "n_random_draws": cfg["n_random_draws"], "goal2_min_gap": cfg["goal2_min_gap"],
    "ceiling_min_pct": cfg["ceiling_min_pct"], "min_pos_frac": cfg["min_pos_frac"], "goal3_episodes": cfg["goal3_episodes"],
    "runtime_min": {k: v / 60 for k, v in st["timings"]["steps"].items()}, "runtime_total_min": st["timings"]["total_s"] / 60,
    "gpu": "A100-SXM4-80GB",
}
S["frames"] = {
    "demo_tasks": sorted(set(cfg["task_ids"]) - set(no_demo)), "rollout_tasks": no_demo,
    "n_rollouts": len(rollouts), "n_rollouts_failed": len(failed), "rollouts_failed": sorted(failed),
    "probe_frames_from_failed_rollouts": probe_from_failed,
    "agentview_bowl_px_median": float(np.median(ag)), "agentview_bowl_px_range": [int(ag.min()), int(ag.max())],
    "agentview_bowl_frac_pct_median": float(100 * np.median(ag) / (H * W)),
    "wrist_bowl_px_median": float(np.median(wr)), "wrist_bowl_frac_pct_median": float(100 * np.median(wr) / (H * W)),
    "agentview_distractor_px_median": float(np.median(dag)),
    "n_distractor_visible": st["frames"]["n_frames_distractor_visible"],
    "n_candidates_rejected": sum(not c["accepted"] for c in J("frames_candidates.json")),
    "n_frames_off_nominal": sum(c["accepted"] and c["t"] != c["t_nominal"] for c in J("frames_candidates.json")),
}

# ------------------------------------------------------------------ checks 1-4
V = J("validate.json")
rows = V["check3"]["rows"]
gap = np.array([r["gap_occluded"] for r in rows])
noise = np.array([r["seed_noise_rmse"] for r in rows])
dist = np.array([r["rmse_occluded_distractor"] for r in rows])
absent = np.array([r["rmse_absent"] for r in rows])
k_td = int((gap > dist).sum())
S["check1"] = {k: V["check1"][k] for k in ("passed", "max_frac_changed_outside", "robot_state_identical", "prompt_tokens_identical",
                                           "absent_render_leftover_px", "n_warnings")}
S["check2"] = {"passed": V["check2"]["passed"], "max_abs_diff": V["check2"]["max_abs_diff"], "n_frames": len(V["check2"]["frames"])}
S["check3"] = {
    "passed": V["check3"]["passed"], "n_frames": len(rows), "median_gap": float(np.median(gap)), "min_gap": float(gap.min()),
    "max_gap": float(gap.max()), "frac_ge_002": float((gap >= 0.02).mean()), "n_ge_01": int((gap >= 0.1).sum()),
    "median_seed_noise": float(np.median(noise)), "gap_over_noise_median": float(np.median(gap / noise)),
    "gap_over_noise_iqr": [float(np.percentile(gap / noise, 25)), float(np.percentile(gap / noise, 75))],
    "frac_gap_gt_noise": float((gap > noise).mean()),
    "median_rmse_by_condition": V["check3"]["median_rmse_by_condition"],
    "target_vs_distractor": {"median_target": float(np.median(gap)), "median_distractor": float(np.median(dist)),
                             "median_ratio": float(np.median(gap / np.maximum(dist, 1e-9))),
                             "n_target_gt_distractor": k_td, "n": len(rows), "sign_test_p": binom_two_sided(k_td, len(rows))},
    "absent_over_occluded_median": float(np.median(absent / gap)),
    "by_split": {s: float(np.median([r["gap_occluded"] for r in rows if r["split"] == s])) for s in ("discovery", "test")},
    "by_phase": {str(p): float(np.median([r["gap_occluded"] for r in rows if r["phase"] == p])) for p in cfg["frame_phases"]},
}
c4 = V["check4"]
rm, mm = np.array(c4["resid_swap_pct_mean"]), np.array(c4["mlp_swap_pct_mean"])
S["check4"] = {"passed": c4["passed"], "n_frames": len(c4["frames"]), "best_layer": c4["best_layer"], "best_pct": c4["best_pct"],
               "resid_swap_mean": rm.tolist(), "mlp_swap_mean": mm.tolist(), "last_layer_ge_90": int(np.nonzero(rm >= 90)[0].max()),
               "first_layer_below_50": int(np.nonzero(rm < 50)[0].min()), "first_layer_below_5": int(np.nonzero(rm < 5)[0].min()),
               "max_single_layer_mlp_swap": float(mm.max()), "argmax_single_layer_mlp_swap": int(mm.argmax()),
               "per_frame_best_range": [float(min(c4["per_frame_best_pct"])), float(max(c4["per_frame_best_pct"]))]}

# ------------------------------------------------------------------ check 5 / transcoders
T = J("train.json")["rows"]
fh, ft = np.array([r["fvu_heldout"] for r in T]), np.array([r["fvu_test_frames"] for r in T])
sp = np.array([r["splice_rmse_over_gap"] for r in T])
S["transcoders"] = {
    "fvu_heldout": fh.tolist(), "fvu_test_frames": ft.tolist(), "splice_over_gap": sp.tolist(),
    "median_fvu": float(np.median(fh)), "median_fvu_test": float(np.median(ft)), "max_fvu": float(fh.max()), "argmax_fvu": int(fh.argmax()),
    "max_fvu_test": float(ft.max()), "argmax_fvu_test": int(ft.argmax()), "min_fvu": float(fh.min()),
    "test_over_heldout_median": float(np.median(ft / fh)), "n_layers_fvu_gt_02": int((fh > 0.2).sum()),
    "n_layers_fvu_test_gt_02": int((ft > 0.2).sum()), "retrained_layers": [r["layer"] for r in T if r["steps"] > cfg["tc_steps"]],
    "n_layers_splice_gt_1": int((sp > 1).sum()), "max_splice": float(sp.max()), "argmax_splice": int(sp.argmax()),
    "median_splice_mid_layers_1_13": float(np.median(sp[1:14])), "alive_min": min(r["alive_features"] for r in T),
    "n_train_tokens": T[0]["n_train_tokens"], "n_heldout_tokens": T[0]["n_heldout_tokens"],
    "seconds_per_layer_mean": float(np.mean([r["seconds"] for r in T])),
}

# ------------------------------------------------------------------ goal 1 (discovery frames)
with open(R / "goal1_feature_table.csv") as fh_:
    tab = list(csv.DictReader(fh_))
P = [r for r in tab if r["pass"] == "True"]
fl = lambda r, k: float(r[k]) if r[k] not in ("", "None") else float("nan")
spec = np.array([fl(r, "specificity") for r in P])
rise = np.array([fl(r, "rise_tokens") for r in P])
docc = np.array([fl(r, "d_occluded") for r in P])
docab = np.array([fl(r, "d_occluded_absent") for r in P])
dratio = np.array([fl(r, "distractor_over_occluded") for r in P])
g1 = st["goal1"]
sel = J("goal1_selected.json")
top_feats = sorted(((s["score"], int(l), s["feature"], s["specificity"], s["rise_tokens"], s["frames_positive"], s["distractor_over_occluded"])
                    for l, v in sel["layers"].items() for s in v["occ_stats"]), reverse=True)
S["goal1"] = {
    "n_features_total": len(tab), "n_alive": sum(r["alive"] == "True" for r in tab), "n_pass": len(P),
    "pass_frac_pct": 100 * len(P) / len(tab), "n_pass_target": g1["n_pass_target_total"], "n_pass_per_layer": g1["n_pass_per_layer"],
    "n_layers_with_pass": sum(v > 0 for v in g1["n_pass_per_layer"].values()), "top_layers": g1["top_layers"],
    "max_per_layer": max(g1["n_pass_per_layer"].values()), "cap": cfg["max_features_per_layer"],
    "specificity_median": float(np.median(spec)), "specificity_range": [float(spec.min()), float(spec.max())],
    "rise_tokens_median": float(np.median(rise)), "rise_tokens_range": [float(rise.min()), float(rise.max())],
    "frames_positive_median": float(np.median([fl(r, "frames_positive") for r in P])),
    "corr_d_occluded_vs_d_occluded_absent": float(np.corrcoef(docc, docab)[0, 1]),
    "ratio_occluded_absent_over_occluded_median": float(np.median(docab / docc)),
    "median_abs_other_over_occ": {c: float(np.median(np.abs([fl(r, f"d_{c}") for r in P]) / docc)) for c in ("recolor", "absent", "slab_miss")},
    "distractor_over_occluded_median": float(np.nanmedian(dratio)), "distractor_over_occluded_max": float(np.nanmax(dratio)),
    "n_discovery_frames": g1["n_discovery_frames"], "n_distractor_frames": g1["n_distractor_frames"],
    "top_features": [{"layer": l, "feature": j, "score": s, "specificity": sp_, "rise": ri, "frames_positive": fp, "distractor_ratio": dr}
                     for s, l, j, sp_, ri, fp, dr in top_feats[:5]],
    "color_set_size": sum(len(v["color"]) for v in sel["layers"].values()),
}

# ------------------------------------------------------------------ goal 2 (held-out test frames)
G = J("goal2_stats.json")
site = G["sites"]["all"]
m = lambda k: site[k]["mean"]
ci = lambda k: site[k]["ci95"]
Pr = G["paired"]
with open(R / "goal2_rows.csv") as fh_:
    g2rows = list(csv.DictReader(fh_))


def per_frame(method, layer="-2", scope="all", direction="inject"):
    by = {}
    for r in g2rows:
        if r["layer"] == layer and r["method"] == method and r["scope"] == scope and r["direction"] == direction:
            by.setdefault(int(r["frame"]), []).append(float(r["closed_pct"]))
    return {k: float(np.mean(v)) for k, v in sorted(by.items())}


occ_pf, rnd_pf = per_frame("occ_features"), per_frame("random_same_norm")
tasks_pf = {}
for r in g2rows:
    tasks_pf[int(r["frame"])] = int(r["task_id"])
fr_ = sorted(occ_pf)
recomputed_occ = float(np.mean([occ_pf[k] for k in fr_]))
d = np.array([occ_pf[k] - rnd_pf[k] for k in fr_])
eq_upper = cluster_boot(d, [tasks_pf[k] for k in fr_], seed=1)[1][1]
kc = G["kcurve"]
em = G["edit_mass_and_overlap"]
S["goal2"] = {
    "n_test_frames": G["protocol"]["n_test_frames"], "n_included": G["protocol"]["n_included"],
    "excluded": [(x["frame"], x["task_id"], x["gap"]) for x in G["protocol"]["excluded"]], "n_tasks": len(G["protocol"]["tasks"]),
    "means": {k: m(k) for k in site}, "ci95": {k: ci(k) for k in site}, "sd": {k: site[k]["sd"] for k in site},
    "recomputed_occ_mean_from_rows": recomputed_occ,
    "occ_over_random": m("occ_features") / m("random_same_norm"),
    "occ_share_of_ceiling_pct": 100 * m("occ_features") / m("mlp_swap_ceiling"),
    "occ_share_of_all_tc_pct": 100 * m("occ_features") / m("all_tc_features"),
    "all_tc_share_of_ceiling_pct": 100 * m("all_tc_features") / m("mlp_swap_ceiling"),
    "paired": Pr, "margin": G["verdict"]["all"]["margin_vs_best_control"], "passed": G["verdict"]["all"]["passed"],
    "margin_required": G["protocol"]["margin_pts"],
    "diff_ci_upper_recomputed_seed1": eq_upper,
    "beats_every_random_draw": G["occ_beats_every_random_draw"],
    "frac_frames_occ_positive": float(np.mean([occ_pf[k] > 0 for k in fr_])),
    "occ_frame_range": [float(min(occ_pf.values())), float(max(occ_pf.values()))],
    "scopes": {sc: {mm_: G["scopes"][sc][mm_]["mean"] for mm_ in G["scopes"][sc]} for sc in G["scopes"]},
    "remove": {k: v["mean"] for k, v in G["remove"].items()}, "remove_ci": {k: v["ci95"] for k, v in G["remove"].items()},
    "single_layers": {l: {k: G["sites"][l][k]["mean"] for k in G["sites"][l]} for l in G["sites"] if l != "all"},
    "single_features": [{"layer": s["layer"], "feature": s["feature"], "mean": s["mean"], "ci95": s["ci95"]} for s in G["single_features"]],
    "kcurve": {fam: {p["k"]: {"mean": p["mean"], "ci95": p["ci95"]} for p in kc[fam]} for fam in ("oracle", "discovery", "random")},
    "kcurve_all_tc": kc["all_tc"]["mean"], "n_total_features": kc["n_total_features"],
    "oracle_over_random_k500": kc["oracle"][3]["mean"] / kc["random"][0]["mean"],
    "oracle_over_random_k2000": kc["oracle"][5]["mean"] / kc["random"][1]["mean"],
    "discovery_over_random_k2000": kc["discovery"][2]["mean"] / kc["random"][1]["mean"],
    "oracle5000_share_of_all_tc_pct": 100 * kc["oracle"][6]["mean"] / kc["all_tc"]["mean"],
    "edit_mass": {k: v["median"] for k, v in em.items()},
    "heldout": G["heldout_selectivity"], "edges": G["edges"],
    "gaps": G["gaps"],
}

# edit-mass composition by layer (discovery ranking = mean over discovery frames of y_scale * sum_tokens |f_occ - f_base|)
rk = torch.load(R / "goal1_rank.pt", weights_only=False)
Ls = sorted(rk["edit"])
E = torch.stack([rk["edit"][l].float() for l in Ls]).numpy()
order = np.argsort(-E.reshape(-1))
lay_of = lambda idx: np.asarray(idx) // E.shape[1]
S["edit_mass"] = {
    "y_scale": {l: float(rk["y_scale"][l]) for l in Ls},
    "share_by_layer_pct": {l: float(100 * E[i].sum() / E.sum()) for i, l in enumerate(Ls)},
    "share_layers_16_17_pct": float(100 * E[16:].sum() / E.sum()), "share_layers_14_17_pct": float(100 * E[14:].sum() / E.sum()),
    "top50_layers": {int(l): int(c) for l, c in zip(*np.unique(lay_of(order[:50]), return_counts=True))},
    "top500_layers": {int(l): int(c) for l, c in zip(*np.unique(lay_of(order[:500]), return_counts=True))},
    "top50_frac_in_16_17": float(np.mean(lay_of(order[:50]) >= 16)),
    "top500_frac_in_14_17": float(np.mean(lay_of(order[:500]) >= 14)),
}

# held-out replication recomputed from the raw per-feature sums (sanity check of goal2_stats)
X = np.load(R / "goal2_test_feature_sums.npy")  # (frames, conds, layers, feats)
Dd = {c: X[:, i] - X[:, 0] for i, c in enumerate(CONDS) if c != "base"}
occ_m = Dd["occluded"].mean(0)
typ = np.stack([rk["typical"][l].numpy() for l in Ls])
alive = np.stack([rk["alive"][l].numpy() for l in Ls]).astype(bool)
other = np.max(np.stack([np.abs(Dd[c]).mean(0) for c in ("recolor", "absent", "slab_miss")]), 0)
with np.errstate(divide="ignore", invalid="ignore"):
    spec_te = np.where(occ_m > 0, 1 - other / np.maximum(occ_m, 1e-12), -np.inf)
rep = (occ_m > 0) & ((Dd["occluded"] > 0).mean(0) >= cfg["min_pos_frac"]) & (spec_te >= cfg["min_specificity"]) \
    & (occ_m / np.maximum(typ, 1e-8) >= cfg["min_rise_tokens"]) & alive
selS = [(int(l), j) for l, v in sel["layers"].items() for j in v["occ"]]
rep_sel = [bool(rep[l, j]) for l, j in selS]
S["heldout_recomputed"] = {
    "n_selected": len(selS), "n_replicate": int(sum(rep_sel)), "frac": float(np.mean(rep_sel)),
    "n_replicating_all_alive": int(rep[alive].sum()), "n_alive": int(alive.sum()),
    "base_rate": float(rep[alive].mean()), "enrichment": float(np.mean(rep_sel) / max(rep[alive].mean(), 1e-12)),
    "selected_test_specificity_median": float(np.median([spec_te[l, j] for l, j in selS])),
    "selected_test_frames_positive_median": float(np.median([(Dd["occluded"][:, l, j] > 0).mean() for l, j in selS])),
    "replicating_features": [f"L{l}f{j}" for (l, j), ok in zip(selS, rep_sel) if ok],
}

# ------------------------------------------------------------------ goal 3 (exploratory closed loop)
g3 = J("goal3.json")
pe = {k: np.array(v) for k, v in g3["per_episode"].items()}
disc = lambda a, b: (int(((pe[a] == 1) & (pe[b] == 0)).sum()), int(((pe[a] == 0) & (pe[b] == 1)).sum()))
S["goal3"] = {"task_id": g3["task_id"], "n_episodes": len(pe["baseline_hidden"]), "success": {k: float(v.mean()) for k, v in pe.items()},
              "stats": g3["stats"], "n_layers": len(g3["layers"]), "n_features": sum(len(v) for v in g3["features"].values()),
              "discordant_baseline_vs_ablated": disc("baseline_hidden", "features_ablated"),
              "discordant_baseline_vs_amplified": disc("baseline_hidden", "features_amplified"),
              "discordant_clean_vs_hidden": disc("clean_reference", "baseline_hidden"),
              "mcnemar_exact_p_ablated": binom_two_sided(0, sum(disc("baseline_hidden", "features_ablated"))),
              "median_steps_success": {k: float(np.median([s for s, ok in zip(g3["steps"][k], g3["per_episode"][k]) if ok])) for k in pe},
              "exploratory": g3["exploratory"]}

# ------------------------------------------------------------------ v1 vs v2
v1 = json.loads((ROOT / "analysis" / "v1_derived_stats.json").read_text())
A1 = v1["goal2"]["all_layers"]
S["v1"] = {
    "n_tasks": 1, "n_frames": v1["setup"]["n_probe_frames"], "n_tokens": v1["setup"]["n_tokens"], "tc_features": v1["setup"]["tc_features"],
    "tc_k": v1["setup"]["tc_k"], "n_features_total": v1["goal1"]["n_features_total"], "median_fvu": v1["transcoders"]["median_fvu"],
    "gap_median": float(np.median(v1["check3"]["gap_occluded"])), "gap_range": [min(v1["check3"]["gap_occluded"]), max(v1["check3"]["gap_occluded"])],
    "gap_over_noise_range": [min(v1["check3"]["gap_over_seed_noise"]), max(v1["check3"]["gap_over_seed_noise"])],
    "check4_best": v1["check4"]["best_pct"], "check4_best_layer": v1["check4"]["best_layer"], "max_single_mlp_swap": v1["check4"]["max_single_layer_mlp_swap"],
    "n_pass": v1["goal1"]["n_pass"], "pass_frac_pct": v1["goal1"]["pass_frac_pct"], "top_layers": v1["goal1"]["top_layers"],
    "specificity_median": v1["goal1"]["specificity_median"], "corr_occ_occabs": v1["goal1"]["corr_d_occluded_vs_d_occluded_absent"],
    "means": A1["means"], "margin": v1["goal2"]["verdict"]["all"]["margin_vs_best_control"], "p": A1["sign_flip_p_vs_random"],
    "occ_share_of_ceiling_pct": 100 * A1["means"]["occ_features"] / A1["means"]["mlp_swap_ceiling"],
    "bowl": {k: float(np.mean(v)) for k, v in A1["bowl"].items()}, "nonbowl": {k: float(np.mean(v)) for k, v in A1["nonbowl"].items()},
    "remove": {k: float(np.mean(v)) for k, v in A1["remove"].items()},
    "frames_beat_every_draw": A1["frames_occ_gt_every_random_draw"], "n_random_draws": 5,
    "top_single_feature": v1["goal2"]["single_features"][0]["mean"],
    "edge": next(e["mean_frac_recovered"] for e in v1["goal2"]["edges"] if e["from_layer"] == 4 and e["to_layer"] == 5 and e["source"] == "occ"),
    "runtime_min": sum(v1["runtime_s"].values()) / 60,
}
S["v1_to_v2"] = {
    "occ_drop_factor": S["v1"]["means"]["occ_features"] / S["goal2"]["means"]["occ_features"],
    "margin_drop_factor": S["v1"]["margin"] / S["goal2"]["margin"],
    "v1_occ_outside_v2_ci": S["v1"]["means"]["occ_features"] > S["goal2"]["ci95"]["occ_features"][1],
}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


(ROOT / "analysis" / "derived_stats.json").write_text(json.dumps(_clean(S), indent=1))
print("wrote analysis/derived_stats.json")
for k in ("frames", "check3", "transcoders", "goal1", "heldout_recomputed", "edit_mass", "goal3", "v1_to_v2"):
    s = json.dumps(_clean(S[k]))
    print(f"== {k}: {s[:1400]}")
print("== goal2 means:", {k: round(v, 3) for k, v in S["goal2"]["means"].items()})
print("== goal2 recomputed occ mean:", S["goal2"]["recomputed_occ_mean_from_rows"], "| ratios", S["goal2"]["occ_over_random"],
      S["goal2"]["occ_share_of_ceiling_pct"], S["goal2"]["all_tc_share_of_ceiling_pct"], "| frac frames occ>0", S["goal2"]["frac_frames_occ_positive"],
      "| occ range", S["goal2"]["occ_frame_range"], "| k-ratios", S["goal2"]["oracle_over_random_k500"], S["goal2"]["oracle_over_random_k2000"],
      S["goal2"]["discovery_over_random_k2000"], S["goal2"]["oracle5000_share_of_all_tc_pct"])


# ------------------------------------------------------------------ robustness of the Goal-2 null (appended block)
def goal2_subset(keep):
    fr = [k for k in sorted(occ_pf) if keep(k)]
    o = np.array([occ_pf[k] for k in fr])
    r = np.array([rnd_pf[k] for k in fr])
    tk = [tasks_pf[k] for k in fr]
    mo, cio = cluster_boot(o, tk)
    md, cid = cluster_boot(o - r, tk)
    return {"n_frames": len(fr), "n_tasks": len(set(tk)), "occ": mo, "occ_ci": cio, "diff": md, "diff_ci": cid,
            "occ_median": float(np.median(o)), "random": float(r.mean())}


failed_ids = set(S["frames"]["probe_frames_from_failed_rollouts"])
gmax = max(G["gaps"]["max"], 0)
big = [x["frame"] for x in G["protocol"]["excluded"]]
gap_by = {int(k): v for k, v in st["validate"]["gap_by_frame"].items()}
S["robustness"] = {
    "all_included": goal2_subset(lambda k: True),
    "without_failed_rollout_frames": goal2_subset(lambda k: k not in failed_ids),
    "without_largest_gap_frame": goal2_subset(lambda k: gap_by[k] < max(gap_by[x] for x in occ_pf)),
    "demo_task0_only": goal2_subset(lambda k: tasks_pf[k] == 0),
    "rollout_tasks_only": goal2_subset(lambda k: tasks_pf[k] != 0),
    "gap_ge_02": goal2_subset(lambda k: gap_by[k] >= 0.2),
}
(ROOT / "analysis" / "derived_stats.json").write_text(json.dumps(_clean(S), indent=1))
print("== robustness:", json.dumps(_clean(S["robustness"]), indent=0)[:2500])


# ------------------------------------------------------------------ reviewer-requested checks (appended block)
rnd_draws = {}
for r in g2rows:
    if r["layer"] == "-2" and r["method"] == "random_same_norm" and r["scope"] == "all" and r["direction"] == "inject":
        rnd_draws.setdefault(int(r["frame"]), []).append(float(r["closed_pct"]))
by_task = {}
for r in rows:
    by_task.setdefault(r["task_id"], []).append((r["gap_occluded"], r["rmse_occluded_distractor"]))
kt = sum(np.mean([a for a, _ in v]) > np.mean([b for _, b in v]) for v in by_task.values())
Oc = X[:, 3] - X[:, 0]
Dc = X[:, 6] - X[:, 0]
ys = np.array([rk["y_scale"][l] for l in Ls])
proxy = [float(np.mean(np.argsort(-(np.abs(Oc[i]) * ys[:, None]).reshape(-1))[:50] // Oc.shape[2] >= 16)) for i in range(Oc.shape[0])]
test_ratio = {f"L{l}f{j}": float(np.abs(Dc[:, l, j]).mean() / Oc[:, l, j].mean()) for l, j in selS}
S["review"] = {
    "median_abs_closure_occ": float(np.median([abs(occ_pf[k]) for k in fr_])),
    "median_abs_closure_random_draw": float(np.median([abs(x) for k in fr_ for x in rnd_draws[k]])),
    "frames_occ_beats_every_draw": int(sum(occ_pf[k] > max(rnd_draws[k]) for k in fr_)),
    "frames_occ_loses_to_every_draw": int(sum(occ_pf[k] < min(rnd_draws[k]) for k in fr_)),
    "expected_beats_every_draw_if_exchangeable": len(fr_) / (len(rnd_draws[fr_[0]]) + 1),
    "tasks_target_gt_distractor": int(kt), "tasks_sign_test_p_two_sided": binom_two_sided(int(kt), len(by_task)),
    "task3_target_gt_distractor_frames": int(sum(a > b for a, b in by_task[3])),
    "wrist_invisible_frames": [p["frame_id"] for p in probe if p["edit_info"]["image2"]["mask_px"] == 0],
    "goal1_pass_under_v1_all_frames_rule": int(sum(float(r["frames_positive"]) == 1.0 for r in P)),
    "check3_under_v1_rule_passes": bool(gap.min() >= 0.02), "check3_min_gap": float(gap.min()),
    "test_distractor_ratio_gt_04": {k: v for k, v in test_ratio.items() if v > 0.4},
    "proxy_test_top50_share_L16_17_median": float(np.median(proxy)), "proxy_test_top50_share_L16_17_min": float(min(proxy)),
    "L16_resid_swap": float(rm[16]), "L16_mlp_swap": float(mm[16]), "L17_share_pct": S["edit_mass"]["share_by_layer_pct"][17],
    "L16_share_pct": S["edit_mass"]["share_by_layer_pct"][16],
    "v1_over_v2": {k: S["v1"]["means"][k] / S["goal2"]["means"][k] for k in S["v1"]["means"]},
    "v1_occ_over_random": S["v1"]["means"]["occ_features"] / S["v1"]["means"]["random_same_norm"],
    "v1_patched_share_pct": 100 * 72 / 9216, "v2_patched_share_pct": 100 * 29 / 36864,
    "n_replicate_fail": len(selS) - int(sum(rep_sel)),
}
(ROOT / "analysis" / "derived_stats.json").write_text(json.dumps(_clean(S), indent=1))
print("== review:", json.dumps(_clean(S["review"]))[:2000])
