# Transcoder circuit tracing on π0.5 (LIBERO), protocol v2: results

Bundle created 2026-09-23 10:08 from `/content/groot-run/outputs/permanence/tc_circuit_v2`. Total run time: 123 min.

Policy `lerobot/pi05_libero_finetuned` · suite `libero_spatial` tasks [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] · 60 probe frames (20 discovery / 40 held-out test) · transcoders 2048 features, k=32, 4000 steps, trained on discovery samples only

## What changed from v1

- 10 tasks instead of 1; 60 probe frames instead of 4.
- Frames sit at fixed phases (0.4, 0.8) of the pre-grasp window instead of being picked for the largest occlusion effect.
- Discovery / test split by demonstration: features and transcoders use demo 0 (+ unedited frames from demos 3-4); the causal test uses demos 1-2 only.
- New control condition `occluded_distractor`: the same gray paint on the second, identical bowl.
- 4x wider transcoders (2048 features, k=32), FVU and splice also measured on held-out test frames.
- Goal 2: 10 random draws, uncapped and target-specific feature sets, a k-curve (how many features it takes), task-clustered bootstrap CIs, sign-flip permutation tests, held-out replication of Goal-1 selectivity.
- Primary Goal-2 site is all layers at once (pre-registered here; it was a post-hoc extension in v1). Same +10 pt rule.
- Goal 3 runs as an explicitly exploratory closed-loop check (10 episodes/condition) when Goal 2 fails.

## Checks and goals

| item | result |
|---|---|
| Check 1: edits are clean | PASS |
| Check 2: determinism | PASS |
| Check 3: the signal exists | PASS (98% of frames ≥ 0.02; median gap 0.186) |
| Check 4: the ceiling works | PASS (97.1% at layer 1; threshold 80.0%) |
| Check 5: transcoder is faithful | PASS (median FVU 0.185 held-out tokens, 0.400 held-out test frames) |
| Goal 1: occlusion features exist | PASS |
| Goal 2: they cause the action (held-out test frames) | FAIL |
| Check 6: closed loop measurable | PASS |
| Goal 3: task success shifts (exploratory) | FAIL |

## Goal 1 (discovery frames)

- 29 of 36864 features pass (0.08%); 29 also stay quiet when the distractor bowl is painted.
- Median distractor/target response ratio among passing features: 0.03.
- Top layers: [5, 1, 7]. Passing features per layer: {'0': 1, '1': 3, '2': 3, '3': 0, '4': 2, '5': 5, '6': 3, '7': 5, '8': 1, '9': 1, '10': 2, '11': 1, '12': 1, '13': 0, '14': 0, '15': 1, '16': 0, '17': 0}.

## Goal 2 (held-out test frames, all layers patched at once)

34 of 40 test frames (gap ≥ 0.1), 10 tasks. Gap closed %, mean [95% task-clustered bootstrap CI].

| method | gap closed % |
|---|---|
| occ_features | 0.27 [-0.32, 0.92] |
| occ_features_target_specific | 0.27 [-0.32, 0.92] |
| occ_features_uncapped | 0.27 [-0.32, 0.92] |
| color_features | 0.05 [-0.10, 0.17] |
| random_same_norm | 0.05 [-0.01, 0.11] |
| random_same_norm_uncapped | 0.03 [-0.10, 0.15] |
| random_raw | 0.05 [-0.11, 0.20] |
| all_tc_features | 42.77 [35.94, 50.64] |
| mlp_swap_ceiling | 76.61 [68.51, 84.78] |

- all_occ_vs_random_same_norm: +0.21 pts [-0.35, +0.84], p(tasks) = 0.222, p(frames) = 0.25, positive in 7/10 tasks
- all_occ_vs_color: +0.22 pts [-0.39, +0.86], p(tasks) = 0.191, p(frames) = 0.25, positive in 7/10 tasks
- all_uncapped_vs_random_uncapped: +0.24 pts [-0.30, +0.86], p(tasks) = 0.199, p(frames) = 0.23, positive in 6/10 tasks
- all_target_specific_vs_random_same_norm: +0.21 pts [-0.35, +0.84], p(tasks) = 0.222, p(frames) = 0.25, positive in 7/10 tasks
- L5_occ_vs_random_same_norm: -0.13 pts [-0.34, +0.07], p(tasks) = 0.789, p(frames) = 0.86, positive in 5/10 tasks
- L1_occ_vs_random_same_norm: -0.08 pts [-0.23, +0.06], p(tasks) = 0.871, p(frames) = 0.85, positive in 3/10 tasks
- L7_occ_vs_random_same_norm: +0.07 pts [-0.06, +0.18], p(tasks) = 0.179, p(frames) = 0.12, positive in 7/10 tasks

**Verdict:** margin +0.21 pts vs the +10 pt rule → FAIL. Occlusion features = 0.3% of the MLP-swap ceiling.

### k-curve

| ranking | K=50 | K=100 | K=200 | K=500 | K=1000 | K=2000 | K=5000 |
|---|---|---|---|---|---|---|---|
| oracle (this frame) | 0.0 | 0.1 | 1.0 | 6.7 | 11.9 | 18.2 | 29.8 |

Discovery-ranked: K=100: 0.0%, K=500: 2.1%, K=2000: 6.5%, K=5000: 10.4% · random K: K=500: 0.6%, K=2000: 3.8% · all 36,864 features: 42.8%

### Other Goal-2 numbers

```json
{
 "edit_mass_and_overlap": {
  "capped_in_oracle_top_n_frac": {
   "median": 0.0,
   "mean": 0.0
  },
  "capped_median_oracle_rank_pct": {
   "median": 13.1591796875,
   "mean": 16.63563048917484
  },
  "capped_share_of_edit_mass_pct": {
   "median": 0.04347608797252178,
   "mean": 0.05409822521238204
  },
  "discovery_vs_oracle_top500_overlap": {
   "median": 0.595,
   "mean": 0.5927647058823529
  },
  "n_features_for_50pct_edit_mass": {
   "median": 35.5,
   "mean": 35.64705882352941
  },
  "n_features_for_80pct_edit_mass": {
   "median": 306.0,
   "mean": 349.5882352941176
  },
  "n_features_for_90pct_edit_mass": {
   "median": 1446.0,
   "mean": 1534.3529411764705
  }
 },
 "heldout_selectivity": {
  "n_test_frames": 40,
  "criteria": "same Goal-1 rule, recomputed on held-out test frames only",
  "replicate_frac_capped": 0.27586206896551724,
  "n_capped": 29,
  "replicate_frac_uncapped": 0.27586206896551724,
  "n_uncapped": 29,
  "replicate_frac_target_specific": 0.27586206896551724,
  "n_target_specific": 29,
  "replicate_frac_color": 0.0,
  "n_color": 34,
  "replicate_frac_random_draws_mean": 0.0,
  "replicate_frac_all_alive_base_rate": 0.0003260692353676431,
  "spearman_d_occluded_discovery_vs_test_all_alive": 0.6504228793556854,
  "pearson_d_occluded_discovery_vs_test_all_alive": 0.7985826941952751,
  "distractor_over_occluded_median_capped": 0.05782536417245865,
  "n_frames_distractor_visible": 40
 },
 "occ_beats_every_random_draw": {
  "count": 12,
  "n": 34
 },
 "scopes": {
  "bowl": {
   "occ_features": {
    "mean": 0.2623549423629732,
    "ci95": [
     -0.20651046903218528,
     0.8275222999092035
    ],
    "sd": 1.8480587281374543,
    "median": 0.4439450827387781,
    "n_frames": 34,
    "n_tasks": 10
   },
   "mlp_swap_ceiling": {
    "mean": 26.696127244231796,
    "ci95": [
     21.94736108245944,
     31.37481425062649
    ],
    "sd": 19.26401752909012,
    "median": 24.608142442287487,
    "n_frames": 34,
    "n_tasks": 10
   }
  },
  "nonbowl": {
   "occ_features": {
    "mean": 0.0901411968338771,
    "ci95": [
     -0.1526878568735716,
     0.31163147132228397
    ],
    "sd": 0.5682310868283094,
    "median": 0.18877803453761222,
    "n_frames": 34,
    "n_tasks": 10
   },
   "mlp_swap_ceiling": {
    "mean": 61.371986736585846,
    "ci95": [
     52.121178585104836,
     71.31438852025143
    ],
    "sd": 28.91028087915414,
    "median": 74.36141103163067,
    "n_frames": 34,
    "n_tasks": 10
   }
  }
 },
 "remove": {
  "occ_features": {
   "mean": 0.5141428385559046,
   "ci95": [
    -1.954879736238664,
    3.5098232294887475
   ],
   "sd": 8.37394049822662,
   "median": -0.050853195982270005,
   "n_frames": 34,
   "n_tasks": 10
  },
  "color_features": {
   "mean": -0.06191347782430565,
   "ci95": [
    -0.1429200851668922,
    0.03633496047878134
   ],
   "sd": 0.44714420132356336,
   "median": -0.05169531804669614,
   "n_frames": 34,
   "n_tasks": 10
  },
  "mlp_swap_ceiling": {
   "mean": 87.99184973057825,
   "ci95": [
    86.52060566160024,
    89.30503456761522
   ],
   "sd": 5.89927613885563,
   "median": 90.00484477768316,
   "n_frames": 34,
   "n_tasks": 10
  }
 },
 "single_features": [
  {
   "layer": 1,
   "feature": 1113,
   "goal1_score": 38.155999163393496,
   "mean": 0.03429178396190783,
   "ci95": [
    -0.06737773082887388,
    0.12705390522375357
   ],
   "sd": 0.34808480807829906,
   "median": 0.10532142014875734,
   "n_frames": 34,
   "n_tasks": 10
  },
  {
   "layer": 5,
   "feature": 93,
   "goal1_score": 31.892732724971566,
   "mean": 0.12063775704725585,
   "ci95": [
    0.026465189158217228,
    0.20886295756901305
   ],
   "sd": 0.3246378356734555,
   "median": 0.12555451250553795,
   "n_frames": 34,
   "n_tasks": 10
  },
  {
   "layer": 1,
   "feature": 564,
   "goal1_score": 30.82558867525743,
   "mean": 0.09439957998769667,
   "ci95": [
    0.03537427011280453,
    0.16028808302990458
   ],
   "sd": 0.3500257693941206,
   "median": 0.0457947986146956,
   "n_frames": 34,
   "n_tasks": 10
  }
 ],
 "edges": [
  {
   "from_layer": 1,
   "to_layer": 5,
   "source": "occ",
   "mean_frac_recovered": 0.006422618492712031,
   "n": 34
  },
  {
   "from_layer": 1,
   "to_layer": 5,
   "source": "random",
   "mean_frac_recovered": 0.00010084789207583445,
   "n": 34
  },
  {
   "from_layer": 1,
   "to_layer": 7,
   "source": "occ",
   "mean_frac_recovered": 0.0029548600252687555,
   "n": 34
  },
  {
   "from_layer": 1,
   "to_layer": 7,
   "source": "random",
   "mean_frac_recovered": -0.00017062965002208916,
   "n": 34
  },
  {
   "from_layer": 5,
   "to_layer": 7,
   "source": "occ",
   "mean_frac_recovered": 0.023595427361165003,
   "n": 34
  },
  {
   "from_layer": 5,
   "to_layer": 7,
   "source": "random",
   "mean_frac_recovered": 0.0029481513211882457,
   "n": 34
  }
 ]
}
```

## Goal 3 (exploratory)

- Task 0, 10 episodes per condition.
- Success: {'clean_reference': 0.9, 'baseline_hidden': 0.9, 'features_ablated': 0.8, 'features_amplified': 0.8}
- Stats: {"features_ablated": {"success": 0.8, "diff_vs_baseline": -0.09999999999999998, "ci95": [-0.30000000000000004, 0.0], "predicted_sign": "+", "ci_excludes_zero_in_predicted_direction": false}, "features_amplified": {"success": 0.8, "diff_vs_baseline": -0.09999999999999998, "ci95": [-0.30000000000000004, 0.0], "predicted_sign": "-", "ci_excludes_zero_in_predicted_direction": false}}

## Files

- `results/`: figures (`fig_*.png`), `goal2_stats.json` (every number above), `goal2_circuit_trace_table.md/.csv`, `goal2_rows.csv` (per frame × method), `goal2_frames.json`, `goal2_test_feature_sums.npy`, `goal1_feature_table.csv`, `goal1_selected.json`, `validate.json`, `train.json`, `status/`, `run_v2.log`.
- `code/`: `tc_occlusion.py`, `config.json`, `notebook.ipynb`.
- `env/`: package versions and GPU.
