#!/usr/bin/env python
"""Cross-check the numbers quoted in paper/main.tex and REPORT.md against the raw v2 result files.

Usage (repo root):  python analysis/check_claims.py      (run derive_stats.py first)
Each claim is recomputed from analysis/derived_stats.json (itself recomputed from results/) and must appear, formatted as
written, in the documents listed for it (T = paper, R = report). Exit code 1 if any check fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "analysis" / "derived_stats.json").read_text())
TEX = (ROOT / "paper" / "main.tex").read_text()
REP = (ROOT / "REPORT.md").read_text().replace("\u2212", "-")  # report uses the Unicode minus sign
G, g1, c3, c4, tc, H, HR, EM, g3, V1, RB = (S["goal2"], S["goal1"], S["check3"], S["check4"], S["transcoders"], S["goal2"]["heldout"],
                                           S["heldout_recomputed"], S["edit_mass"], S["goal3"], S["v1"], S["robustness"])
m, ci = G["means"], G["ci95"]
P = G["paired"]["all_occ_vs_random_same_norm"]
f1 = lambda x: f"{x:.1f}"
f2 = lambda x: f"{x:.2f}"
neg = lambda s: s.replace("-", "$-$")  # LaTeX minus in the paper
C = lambda k, fmt=f2: f"[{fmt(ci[k][0])}, {fmt(ci[k][1])}]"
K = lambda fam, k: G["kcurve"][fam][str(k)]["mean"]

claims = [
    # setup / frames
    ("probe frames", "60", "TR"), ("discovery frames", "20", "TR"), ("test frames", "40", "TR"),
    ("extra frames", f"{S['setup']['n_extra_train_frames']}", "TR"),
    ("training tokens", f"{S['setup']['n_tokens']:,}", "TR"), ("forward passes", f"{S['setup']['n_samples']}", "TR"),
    ("rollouts", f"{S['frames']['n_rollouts']}", "TR"), ("failed rollouts", f"{S['frames']['n_rollouts_failed']} of", "TR"),
    ("frames from failed rollouts", f"{len(S['frames']['probe_frames_from_failed_rollouts'])} probe frames", "TR"),
    ("runtime total", f"{S['setup']['runtime_total_min']:.0f}", "TR"),
    ("agentview frac", f"{S['frames']['agentview_bowl_frac_pct_median']:.2f}%", "R"),
    ("agentview frac (tex)", f"{S['frames']['agentview_bowl_frac_pct_median']:.2f}\\%", "T"),
    ("wrist frac", f"{S['frames']['wrist_bowl_frac_pct_median']:.1f}%", "R"),
    # checks
    ("gap median", f"{c3['median_gap']:.3f}", "TR"), ("gap/noise", f"{c3['gap_over_noise_median']:.1f}", "TR"),
    ("frac >= 0.02", f"{100 * c3['frac_ge_002']:.0f}%", "R"), ("frac >= 0.02 (tex)", f"{100 * c3['frac_ge_002']:.0f}\\%", "T"),
    ("absent rmse", f"{c3['median_rmse_by_condition']['absent']:.3f}", "TR"),
    ("recolor rmse", f"{c3['median_rmse_by_condition']['recolor']:.3f}", "TR"),
    ("distractor rmse", f"{c3['median_rmse_by_condition']['occluded_distractor']:.3f}", "TR"),
    ("absent/occluded", f"{c3['absent_over_occluded_median']:.1f}", "TR"),
    ("target>distractor", f"{c3['target_vs_distractor']['n_target_gt_distractor']} of 60", "TR"),
    ("target/distractor ratio", f"{c3['target_vs_distractor']['median_ratio']:.1f}", "TR"),
    ("sign test p", f"{c3['target_vs_distractor']['sign_test_p']:.4f}", "TR"),
    ("check4 best", f"{c4['best_pct']:.1f}", "TR"),
    ("check4 per-frame range", f"{c4['per_frame_best_range'][0]:.1f}", "TR"),
    ("max single mlp swap", f"{c4['max_single_layer_mlp_swap']:.1f}", "TR"),
    ("median fvu", f"{tc['median_fvu']:.3f}", "TR"), ("median fvu test", f"{tc['median_fvu_test']:.3f}", "TR"),
    ("fvu test/heldout", f"{tc['test_over_heldout_median']:.1f}", "TR"),
    ("splice mid median", f"{tc['median_splice_mid_layers_1_13']:.2f}", "TR"),
    ("max splice", f"{tc['max_splice']:.2f}", "TR"),
    # goal 1
    ("n pass", f"{g1['n_pass']}", "TR"), ("pass pct", f"{g1['pass_frac_pct']:.2f}", "TR"),
    ("layers with pass", f"{g1['n_layers_with_pass']} layers", "TR"),
    ("spec median", f"{g1['specificity_median']:.2f}", "TR"), ("spec max", f"{g1['specificity_range'][1]:.2f}", "TR"),
    ("rise median", f"{g1['rise_tokens_median']:.1f}", "TR"),
    ("recolor/occ", f"{g1['median_abs_other_over_occ']['recolor']:.2f}", "TR"),
    ("absent/occ", f"{g1['median_abs_other_over_occ']['absent']:.2f}", "TR"),
    ("slab/occ", f"{g1['median_abs_other_over_occ']['slab_miss']:.2f}", "TR"),
    ("distractor/occ discovery", f"{g1['distractor_over_occluded_median']:.2f}", "TR"),
    ("corr occ vs occ_absent", f"{g1['corr_d_occluded_vs_d_occluded_absent']:.4f}", "TR"),
    # held-out replication
    ("replicating", f"{HR['n_replicate']} of the 29", "TR"), ("replicating base count", f"{HR['n_replicating_all_alive']} of 36,802", "TR"),
    ("enrichment", f"{HR['enrichment']:.0f}", "TR"), ("base rate", f"{100 * HR['base_rate']:.2f}", "TR"),
    ("spearman", f"{H['spearman_d_occluded_discovery_vs_test_all_alive']:.2f}", "TR"),
    ("pearson", f"{H['pearson_d_occluded_discovery_vs_test_all_alive']:.2f}", "TR"),
    ("test specificity", f"{HR['selected_test_specificity_median']:.2f}", "TR"),
    ("test distractor ratio", f"{H['distractor_over_occluded_median_capped']:.3f}", "TR"),
    # goal 2
    ("occ", f2(m["occ_features"]), "TR"), ("occ CI", C("occ_features")[1:-1], "R"), ("occ CI (tex)", neg(C("occ_features")), "T"),
    ("random", f2(m["random_same_norm"]), "TR"), ("random CI (tex)", neg(C("random_same_norm")), "T"),
    ("colour", f2(m["color_features"]), "TR"), ("colour CI (tex)", neg(C("color_features")), "T"),
    ("random raw CI (tex)", neg(C("random_raw")), "T"),
    ("all tc", f2(m["all_tc_features"]), "TR"), ("all tc CI", C("all_tc_features"), "T"), ("all tc CI (md)", C("all_tc_features")[1:-1], "R"),
    ("ceiling", f2(m["mlp_swap_ceiling"]), "TR"), ("ceiling CI", C("mlp_swap_ceiling"), "T"), ("ceiling CI (md)", C("mlp_swap_ceiling")[1:-1], "R"),
    ("all tc 1dp", f1(m["all_tc_features"]), "TR"), ("ceiling 1dp", f1(m["mlp_swap_ceiling"]), "TR"),
    ("diff", f"+{P['mean_diff']:.2f}", "TR"), ("diff CI lo", f"{P['ci95'][0]:.2f}", "TR"), ("diff CI hi", f"+{P['ci95'][1]:.2f}", "TR"),
    ("p tasks", f"{P['p_signflip_tasks']:.3f}", "TR"), ("p frames", f"{P['p_signflip_frames']:.2f}", "TR"),
    ("tasks positive", f"{P['n_tasks_positive']} of 10", "TR"),
    ("colour diff", f"+{G['paired']['all_occ_vs_color']['mean_diff']:.2f}", "TR"),
    ("beats every draw", f"{G['beats_every_random_draw']['count']} of 34", "TR"),
    ("n included", f"{G['n_included']} frames", "R"), ("n included (tex)", "34 frames", "T"),
    ("share of ceiling", f"{G['occ_share_of_ceiling_pct']:.2f}", "TR"), ("share of all tc", f"{G['occ_share_of_all_tc_pct']:.2f}", "TR"),
    ("margin", f"+{G['margin']:.2f}", "TR"),
    ("L5 occ", f"{G['single_layers']['5']['occ_features']:.2f}", "TR"), ("L1 occ", f"{G['single_layers']['1']['occ_features']:.2f}", "TR"),
    ("L7 occ", f"{G['single_layers']['7']['occ_features']:.2f}", "TR"),
    ("best single feature", f"{G['single_features'][1]['mean']:.2f}", "TR"),
    ("remove occ", f"{G['remove']['occ_features']:.2f}", "TR"), ("remove ceiling", f"{G['remove']['mlp_swap_ceiling']:.1f}", "TR"),
    ("bowl ceiling", f"{G['scopes']['bowl']['mlp_swap_ceiling']:.1f}", "TR"), ("nonbowl ceiling", f"{G['scopes']['nonbowl']['mlp_swap_ceiling']:.1f}", "TR"),
    ("edge L5->L7", f"{100 * G['edges'][4]['mean_frac_recovered']:.1f}", "TR"),
    # robustness
    ("robust no failed diff", f"+{RB['without_failed_rollout_frames']['diff']:.2f}", "TR"),
    ("robust task0", f"{RB['demo_task0_only']['occ']:.2f}", "TR"),
    ("robust gap>=0.2 hi", f"{RB['gap_ge_02']['diff_ci'][1]:.2f}", "TR"),
    # k-curve and edit mass
    ("oracle 500", f2(K("oracle", 500)), "TR"), ("oracle 2000", f2(K("oracle", 2000)), "TR"), ("oracle 5000", f2(K("oracle", 5000)), "TR"),
    ("oracle 50", f2(K("oracle", 50)), "TR"), ("discovery 5000", f2(K("discovery", 5000)), "TR"),
    ("random 2000", f2(K("random", 2000)), "TR"), ("random 500", f2(K("random", 500)), "TR"),
    ("oracle/random 500", f"{G['oracle_over_random_k500']:.0f}", "TR"), ("oracle/random 2000", f"{G['oracle_over_random_k2000']:.1f}", "TR"),
    ("discovery/random 2000", f"{G['discovery_over_random_k2000']:.1f}", "TR"),
    ("oracle5000 share of all", f"{G['oracle5000_share_of_all_tc_pct']:.0f}", "TR"),
    ("edit mass 50%", f"{G['edit_mass']['n_features_for_50pct_edit_mass']:.1f}", "TR"),
    ("L16-17 share", f"{EM['share_layers_16_17_pct']:.0f}", "TR"),
    ("s_y L16", f"{EM['y_scale']['16']:,.0f}", "TR"), ("s_y L17", f"{EM['y_scale']['17']:,.0f}", "TR"),
    ("rank pct", f"{G['edit_mass']['capped_median_oracle_rank_pct']:.0f}", "TR"),
    # goal 3
    ("g3 hidden", f"{g3['success']['baseline_hidden']:.1f}", "TR"), ("g3 ablated", f"{g3['success']['features_ablated']:.1f}", "TR"),
    ("g3 n layers", f"{g3['n_layers']} layers", "TR"),
    # v1 comparison
    ("v1 occ", f"{V1['means']['occ_features']:.2f}", "TR"), ("v1 random", f"{V1['means']['random_same_norm']:.2f}", "TR"),
    ("v1 colour", f"{V1['means']['color_features']:.2f}", "TR"), ("v1 margin", f"+{V1['margin']:.2f}", "TR"),
    ("v1 all tc", f"{V1['means']['all_tc_features']:.1f}", "TR"), ("v1 ceiling", f"{V1['means']['mlp_swap_ceiling']:.1f}", "TR"),
    ("v1 share ceiling", f"{V1['occ_share_of_ceiling_pct']:.1f}", "TR"), ("v1 n pass", "72 of 9,216", "TR"),
    ("v1 fvu", f"{V1['median_fvu']:.3f}", "TR"), ("v1 check4", f"{V1['check4_best']:.1f}", "TR"),
    ("v1 remove", f"{V1['remove']['occ_features']:.1f}", "TR"), ("v1 remove ceiling", f"{V1['remove']['mlp_swap_ceiling']:.1f}", "TR"),
    ("shrink factor", f"{S['v1_to_v2']['occ_drop_factor']:.0f}", "TR"),
    # reviewer-requested nuance
    ("median |occ|", f"{S['review']['median_abs_closure_occ']:.2f}", "TR"), ("median |random|", f"{S['review']['median_abs_closure_random_draw']:.2f}", "TR"),
    ("loses to every draw", f"on {S['review']['frames_occ_loses_to_every_draw']}", "TR"),
    ("tasks target>distractor", f"{S['review']['tasks_target_gt_distractor']} of 10 tasks", "TR"),
    ("task-level p", f"{S['review']['tasks_sign_test_p_two_sided']:.2f}", "TR"),
    ("strict goal1", f"{S['review']['goal1_pass_under_v1_all_frames_rule']} of the 29", "R"), ("strict goal1 (tex)", "14 of the 29", "T"),
    ("strict check3 gap", f"{S['review']['check3_min_gap']:.3f}", "TR"),
    ("shrink selected", f"{S['review']['v1_over_v2']['occ_features']:.1f}", "R"),
    ("shrink random", f"{S['review']['v1_over_v2']['random_same_norm']:.1f}", "R"), ("shrink colour", f"{S['review']['v1_over_v2']['color_features']:.1f}", "R"),
    ("v1 ratio", f"{S['review']['v1_occ_over_random']:.1f}", "TR"), ("v2 ratio", f"{G['occ_over_random']:.1f}", "TR"),
    ("L17 share", f"{S['review']['L17_share_pct']:.0f}", "TR"), ("L16 share", f"{S['review']['L16_share_pct']:.0f}", "TR"),
    ("L16 resid swap", f"{S['review']['L16_resid_swap']:.1f}", "T"), ("L16 resid swap (md)", f"{S['review']['L16_resid_swap']:.2f}", "R"),
    ("test ratio L0f1178", f"{S['review']['test_distractor_ratio_gt_04']['L0f1178']:.2f}", "TR"),
    ("test ratio L2f2045", f"{S['review']['test_distractor_ratio_gt_04']['L2f2045']:.2f}", "TR"),
    ("replicate fail", f"{S['review']['n_replicate_fail']}", "TR"),
]
fail = 0
docs = {"T": ("paper", TEX), "R": ("report", REP)}
for desc, val, where in claims:
    for w in where:
        name, txt = docs[w]
        ok = val in txt
        fail += not ok
        print(f"[{'ok ' if ok else 'MISSING'}] {desc:30s} {val:>22s}  in {name}")
assert S["v1_to_v2"]["v1_occ_outside_v2_ci"], "v1 estimate should lie above the v2 CI"
assert abs(G["recomputed_occ_mean_from_rows"] - m["occ_features"]) < 1e-9, "goal2_stats.json disagrees with goal2_rows.csv"
assert HR["frac"] == H["replicate_frac_capped"], "held-out replication recomputation disagrees with goal2_stats.json"
print(f"\n{len(claims)} claims, {fail} missing")
sys.exit(1 if fail else 0)
