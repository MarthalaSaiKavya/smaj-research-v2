# Occlusion features in π0.5, protocol v2: a held-out transcoder circuit analysis

**Research report.** This covers the v2 run in this repository (`results/`, run on 2026-09-23, A100-SXM4-80GB, 123 min). The v1 run it replaces is in [smaj-research](https://github.com/MarthalaSaiKavya/smaj-research). Every number below is recomputed from the raw files by `analysis/derive_stats.py` and checked against this text by `analysis/check_claims.py`.

---

## TL;DR

- **Sparse occlusion features exist.** On 20 discovery frames from 10 LIBERO-Spatial tasks, 29 of 36,864 transcoder features (0.08%) respond selectively when the target bowl is painted over. On discovery frames, all 29 stay quiet when the identical *distractor* bowl is painted (median distractor/target response 0.03).
- **Their selectivity partly replicates out of sample.** On 40 held-out test frames (disjoint demonstrations of the same 10 tasks) that neither the transcoders nor the selection ever saw:
  - 8 of the 29 pass the same selectivity rule again and 21 do not. Only 12 of all 36,802 live features pass it there (base rate 0.03%), so the selection is still enriched **846×**.
  - Discovery and test occlusion responses correlate at Spearman ρ = 0.65 across all features.
- **They do not drive the action.** Patching all 29 into the unoccluded run at all 18 layers closes **0.27% [95% CI −0.32, 0.92]** of the base→occluded action gap. Matched random features close 0.05% and colour features 0.05%.
  - The pre-specified rule needs a +10-point margin. The observed margin is **+0.21** (CI of the difference −0.35 to +0.84; sign-flip p = 0.22 over tasks).
  - The CI rules out the pre-specified effect size by more than 10×.
  - Per frame, the selected features move the action more than random features do (median |gap closed| 0.75 vs 0.22 points), but not consistently toward the occluded action: they beat every random draw on 12 of 34 frames and lose to every draw on 8.
- **The effect is distributed.**
  - All 36,864 features close 42.8% and the full MLP outputs 76.6%.
  - The 5,000 features with the largest activation change close 29.8%, and the top 50 close nothing.
  - 73% of the occlusion-induced activation change sits in layer 17, whose output cannot reach the action expert, and another 7% in layer 16, which barely can.
- **v1's apparent above-chance effect is not confirmed.**
  - v1 (4 frames picked for the largest effect, with features selected and tested on the same frames) found 3.92% vs 0.56% for random, 6.9×, p = 0.125.
  - v2 finds 4.9×, but the paired difference is +0.21 points [−0.35, +0.84].
  - The absolute numbers are not comparable. The controls shrank almost as much as the selected set (random 10.5×, colour 14.4×, selected 14.7×), because v2 patches a much smaller share of a wider dictionary (29 of 36,864 vs 72 of 9,216).
- **Closed loop (exploratory, n = 10 per condition).**
  - Painting the bowl in both cameras on every step leaves success unchanged (0.9 vs 0.9 clean). In the recorded episode the paint keeps the bowl's silhouette, and the robot grasps the gray blob.
  - Ablating or amplifying the 29 features gives 0.8 each, from a single discordant episode. There is no detectable shift (McNemar p = 1.0).

---

## 1. Problem statement

A vision-language-action (VLA) policy has to do something sensible when the object named in its instruction is hidden. We ask how π0.5, a flow-matching VLA built on PaliGemma (SigLIP + Gemma-2B), registers that its target is not visible, and whether that representation drives its actions. The questions are the same as v1:

- **Q1 (Goal 1).** Does π0.5's language backbone contain sparse transcoder features that respond selectively to the target being occluded? "Selectively" means not to recolouring, removal, or the same paint placed elsewhere.
- **Q2 (Goal 2).** Do those features cause the change in the predicted action?
- **Q3 (Goal 3).** Does intervening on them change closed-loop task success?

**Why a v2.** v1 answered Q1 yes and Q2 "weakly". The selected features closed 3.9% of the gap, 6.9× matched random features, but below the +10-point bar. v1's evidence had eight structural weaknesses that a reviewer could use to dismiss either reading:

| # | Weakness in v1 | Consequence |
|---|---|---|
| 1 | One task, 4 probe frames | Sign-flip p could not go below 0.0625; no confidence intervals |
| 2 | Frames picked for the largest base-vs-occluded gap | Selection on the outcome |
| 3 | Features selected and tested on the same 4 frames | Circular; effect sizes inflated |
| 4 | Transcoders trained on the test frames | Faithfulness measured in-sample |
| 5 | No control for "a bowl is painted" vs "the target is hidden" | Target specificity untested |
| 6 | At most 8 features per layer | Could not tell sparse from capped |
| 7 | "Distributed" asserted, not measured | No curve of effect vs number of features |
| 8 | All-layers site added post hoc; closed loop never run | Weak confirmatory status; no behavioural number |

v2 changes the protocol to remove each of these and re-runs every step.

---

## 2. Novelty and contributions

1. **To our knowledge, the first held-out circuit analysis of a VLA.** Features are selected, and transcoders trained, on one set of demonstrations and tested on disjoint ones. This includes a test of whether feature *selectivity* replicates out of sample. It partly does: 8 of 29 features pass again, 846× the base rate, ρ = 0.65. Prior VLA interpretability work (SAEs, probes, steering, attention knockout) evaluates in sample or does not trace MLP features at all.
2. **A distractor-controlled occlusion protocol.** Painting the identical second bowl separates "the target is hidden" from "a bowl-shaped patch appeared".
   - At the action level, painting the target moves π0.5's action more than painting the distractor: median ratio 2.5×, on 44 of 60 frames and in 8 of 10 tasks.
   - At the feature level, all 29 selected features are target-specific.
3. **A measured picture of distributedness.** The k-curve shows how much of the gap the top-K features close, with three rankings (per-frame activation change, discovery-frame change, random). Activation-change rankings are dominated by the last layers, where the MLP output's RMS norm reaches 1,608 (L16) and 7,697 (L17) ("massive activations"). L17's output has no path to the action and L16's almost none. Magnitude-based feature rankings are therefore misleading in this architecture.
4. **A direct in-sample vs held-out comparison.** It asks the same questions under the same success rule. The in-sample pilot's above-chance ratio does not hold up as a significant difference on held-out data. It also shows that absolute patching effects depend on set size and dictionary width, so they must be matched before runs are compared.
5. **A closed-loop check** on the same features, labelled exploratory because the pre-specified gate would have skipped it.

---

## 3. Methodology

### 3.1 Model and intervention sites

- **Policy.** `lerobot/pi05_libero_finetuned` on LeRobot 0.6.1, bfloat16, with `torch.compile` disabled so that hooks run eagerly.
- **Prefix.** Three image slots of 256 SigLIP tokens each (agentview, wrist, and an empty third camera that is masked out), plus up to 200 language tokens.
- **Backbone and action expert.** The prefix runs through an 18-layer Gemma-2B decoder. A 300M action expert integrates a 50×7 action chunk with 10 Euler flow-matching steps, attending to the prefix key/value cache at every layer.
- **Hooks.** For each layer we hook the MLP input, the MLP output and the layer's residual output, on every valid prefix token.

### 3.2 Probe frames and the discovery/test split (new in v2)

- **Tasks.** All 10 LIBERO-Spatial tasks. In each, the target is `akita_black_bowl_1` (from the task's `obj_of_interest`) and the distractor is the identical `akita_black_bowl_2`.
- **Episodes per task.** Demo 0 is *discovery*, demos 1–2 are *test*, and demos 3–4 supply *extra* unedited training frames only.
- **Frame choice.** Frames sit at fixed phases 0.4 and 0.8 of the pre-grasp window [t = 5, lift − 3]. The lift is detected from the target's height.
  - The only search is for visibility: the nearest step within ±6 where the target covers at least 80 px of the agentview image.
  - **Frames are never chosen by the size of the effect.**
- **Result.** 60 probe frames: 20 discovery and 40 test, 6 per task. Another 118 unedited frames come from demos 0, 3 and 4.
- **Demo files.** Only task 0 had demonstration files in the Colab dataset archive. Tasks 1–9 therefore used 45 policy rollouts from init states 0–4 (same role mapping).
  - 7 of the 45 rollouts failed: task 4 inits 1–2, task 5 inits 2–4, task 8 inits 0 and 4.
  - 8 probe frames come from failed rollouts. §4.6 shows the result without them.

### 3.3 Seven conditions

All conditions are rendered from MuJoCo segmentation masks in both cameras. Robot state, prompt and flow-matching noise are identical across conditions.

| Condition | Edit |
|---|---|
| `base` | unedited render |
| `recolor` | target pixels shaded red, luminance preserved |
| `absent` | target moved out of the scene and re-rendered, composited within the mask dilated by 6 px (removes its shadow) |
| `occluded` | target mask dilated by 2 px, filled flat gray (128) |
| `slab_miss` | the same gray shape translated so it touches **neither bowl**; v1 only avoided the target |
| `occluded_absent` | `absent` + the same paint |
| `occluded_distractor` | **new:** the same paint on the distractor bowl, never touching the target |

The target covers a median of 577 px (0.88%) of the agentview image (range 89–1,273) and 5,214 px (8.0%) of the wrist image. The distractor is visible in all 60 frames.

**Action gap.** For a frame with fixed noise, g = RMSE(a_base, a_occ) over the normalised 50×7 chunk. A patch applied to the base run yields a′, and **gap closed = 100·(1 − RMSE(a′, a_occ)/g)**. The *remove* direction patches the occluded run toward base.

### 3.4 Validity battery (checks 1–5), run before any feature claim

| Check | Criterion | v2 change |
|---|---|---|
| 1. Edits clean | < 0.2% of pixels changed outside the declared region; state and prompt identical; no leftover target pixels in `absent` | 60 frames × 6 edits × 2 cameras |
| 2. Determinism | repeated forward pass bit-identical | 8 frames spread over tasks |
| 3. Signal exists | gap ≥ 0.02 on ≥ 90% of frames | was "every frame" (on 4 frames) |
| 4. Ceiling works | a single-layer full residual swap closes ≥ 80% (mean over 8 frames) | was 90% on 4 max-gap frames |
| 5. Transcoders faithful | median held-out FVU ≤ 0.20 | + FVU and splice on 8 held-out **test** frames |

Three thresholds were adapted from v1 to the larger design and fixed before any v2 data existed:
- check 3: every frame → ≥ 90% of frames;
- check 4: 90% → 80%;
- Goal 1 positivity: every frame → ≥ 90% of frames.

These were pre-specified in the plan, not registered with a timestamp. Under v1's strict rules:
- check 4 would still pass (97.1%);
- check 3 would fail, because one frame (gap 0.015) is below 0.02; the Goal-2 gap rule excludes that frame anyway;
- Goal 1 would keep 14 of the 29 features.

### 3.5 Transcoders

- **Architecture.** One TopK transcoder per layer, mapping MLP input to MLP output: 2,048 features, k = 32, unit-norm decoder rows, and inputs and outputs mean-centred and scaled to unit mean norm, so the loss equals FVU.
- **Training.** AuxK (k = 64, weight 1/32), Adam at 2e-3, batch 1,024, 4,000 steps. A layer is retrained once with 8,000 steps if its FVU exceeds 0.2.
- **Data.** 146,599 valid prefix tokens from 258 forward passes on **non-test** demonstrations only: 20 discovery frames × 7 conditions, plus 118 extra frames from demos 0, 3 and 4. 131,940 tokens are for training and 14,659 are held out.

### 3.6 Goal 1: selection (discovery frames only)

For feature j, A_j(c) is its activation summed over valid tokens under condition c, and Δ_j(c) = A_j(c) − A_j(base). A feature passes if:

1. Δ_j(occluded) > 0 on ≥ 90% of the 20 discovery frames;
2. its specificity 1 − max over {recolor, absent, slab_miss} of mean|Δ_j| / mean Δ_j(occluded) is ≥ 0.5 (the v1 rule, unchanged);
3. its rise exceeds one typical token-level firing.

- **Target-specific subset.** Features that also keep specificity ≥ 0.5 once `occluded_distractor` is added to the quiet set.
- **Sets.** Capped at 8 per layer, and also used uncapped.
- **Colour control.** A colour set is ranked the same way with `recolor` as the target, with the same number of features per layer and at least one per layer (34 features in total).
- **Discovery ranking.** Every feature also gets a discovery "activation-change" score: s_y · Σ_tokens |f_occ − f_base|, averaged over discovery frames.

### 3.7 Goal 2: causal test (held-out test frames only)

The feature patch is **y′ = y + s_y Σ_{j∈S} (f_j^target − f_j(x)) d_j**, which keeps the transcoder's error term.

- **Primary site (fixed in advance for v2).** All 18 layers patched at once, all valid tokens, inject direction.
- **Pass rule (unchanged, pre-specified).** Occlusion ≥ random + 10 pts *and* ≥ colour + 10 pts.
- **Controls.** 10 draws of norm-matched random features, 2 raw random draws, the colour set, all transcoder features, and the MLP-output swap (the site's ceiling).
- **Also measured:**
  - scopes (bowl tokens vs the rest) and the remove direction;
  - the top-3 Goal-1 layers individually and the top-3 single features;
  - cross-layer edges;
  - the **k-curve**: top-K features by this frame's own activation change (K = 50…5,000), top-K by the discovery ranking (K = 100…5,000), and random K (500, 2,000).
- **Held-out replication.** The Goal-1 rule is recomputed on the test frames from the summed feature activations under all 7 conditions.
- **Exclusion rule (pre-specified).** Test frames with gap < 0.1 are excluded from patching (6 of 40).
- **Statistics:**
  - 95% percentile bootstrap CIs that resample *tasks* (10,000 draws);
  - Goal-2 p-values: one-sided sign-flip permutation tests over tasks (exact, 2¹⁰ = 1,024 sign patterns) and over frames (Monte Carlo, 200,000 draws);
  - the target-vs-distractor sign test and the Goal-3 McNemar test are two-sided exact binomial tests;
  - how often the occlusion set beats *every* random draw on a frame.

### 3.8 Goal 3: closed loop (exploratory)

- **Setup.** Task 0, 10 init states, and 4 conditions sharing seeds: clean reference; bowl painted gray in both cameras on every step ("hidden"); hidden with the 29 features ablated (×0) at their 13 layers; hidden with them amplified (×3).
- **Statistics.** Paired bootstrap CI vs hidden.
- **Status.** Goal 2 failed, so the pre-specified gate would skip this. It ran because the run's config set `GOAL3_EXPLORATORY=True`, and it is reported as exploratory.

---

## 4. Results

### 4.1 Run summary

One process ran all seven steps in 122.8 min:

| Step | Time (min) |
|---|---|
| frames (policy load and 45 rollouts included) | 41.3 |
| validate | 8.9 |
| capture | 3.9 |
| train | 11.1 |
| Goal 1 | 0.5 |
| Goal 2 | 28.0 |
| Goal 3 | 29.1 |

- **Capture.** The pipeline estimated about 20.3 GB of activations, against 153.6 GB of available RAM, so no tokens were subsampled.
- **Frames.** 59 of 60 sit exactly at their nominal phase; one (task 5, test) moved 2 steps for visibility.

### 4.2 The instrument is sound (checks 1–5 all pass)

| Check | Result |
|---|---|
| 1. Edits clean | 0.000% of pixels changed outside regions (720 image checks); state and prompt identical; 0 leftover px; 0 warnings |
| 2. Determinism | max \|Δa\| = 0.0 on 8 frames |
| 3. Signal exists | 59/60 frames (98%) ≥ 0.02. Median gap 0.186 (range 0.015–1.79), median 2.1× the sampling-noise RMSE (IQR 1.4–3.1×); gap > noise on 85% of frames |
| 4. Ceiling works | residual swap closes **97.1%** at layer 1 (per-frame best 96.4–98.5%) |
| 5. Transcoders | median held-out FVU **0.185** (0.021–0.290). On held-out **test frames** the median is **0.400** (15 of 18 layers above 0.2) |

What the action responds to (median action RMSE vs base over 60 frames):

| Condition | Median action RMSE vs base |
|---|---|
| `absent` | 0.649 |
| `occluded` | 0.186 |
| `occluded_absent` | 0.184 |
| `recolor` | 0.136 |
| `slab_miss` | 0.122 |
| `occluded_distractor` | 0.084 |

- **Removal moves the action most**, 2.8× more than paint (median per-frame ratio).
- **The action tracks the target, not "a bowl".** Painting the target moves the action more than painting the identical distractor on 44 of 60 frames, with a median ratio of 2.5.
  - Frame-level sign test: p = 0.0004, but it treats frames as independent.
  - Task level: 8 of 10 tasks, two-sided p = 0.11.
  - Task 3 goes the other way on all 6 of its frames.
- **Phase.** Gaps are larger at phase 0.8 (0.225) than at 0.4 (0.163), and similar in discovery (0.189) and test (0.176) frames.

**Transcoders generalise poorly out of sample.**
- Test-frame FVU is 1.9× the held-out-token FVU (median). Splicing a mid-layer transcoder (layers 1–13) into the model moves the test-frame action by a median 0.82 of the occlusion gap, and by more than the gap in 4 layers (max 1.22 at L12).
- Six layers (6 and 8–12) stayed above FVU 0.2 even after the 8,000-step retry, which improved them by only about 0.003.
- The v1 FVU (0.174) was measured on held-out tokens of the *same* frames, so it was optimistic in the same way.
- Because patching keeps the error term, the interventions below are exact edits along the dictionary directions. The poor fit limits how much of the MLP computation the dictionary can express, not the validity of the patch.

### 4.3 Where the signal lives

- **Residual stream.** A full residual swap at one layer closes **≥ 90% through layer 7**, drops below 50% from layer 10 and below 5% from layer 14. At L17 it closes exactly 0.0%, because the last layer's output never reaches the action expert, which reads only the per-layer K/V.
- **Single MLP outputs** carry more than in v1, up to 22.1% at L12 (v1: 6.5%), but still far less than the residual stream.

### 4.4 Goal 1: occlusion-selective features exist (PASS)

- **Count.** 29 of 36,864 features pass (0.079%). The v1 dictionary gave 72 of 9,216 (0.78%).
- **Layers.** 13 layers contribute, at most 5 per layer: L0 1, L1 3, L2 3, L4 2, L5 5, L6 3, L7 5, L8 1, L9 1, L10 2, L11 1, L12 1, L15 1. The top layers by score are 5, 1 and 7.
- **The cap never binds,** so the capped, uncapped and target-specific sets are the same 29 features.
- **Selectivity.**
  - Median specificity 0.64 (0.50–0.89); median rise 29.0 token-firings (4.7–73.2); median positive on 95% of discovery frames.
  - Relative to the occlusion response: recolor 0.20, absent 0.16, slab_miss 0.09.
  - **Distractor 0.03** (max 0.34, L0 f1178): all 29 are target-specific.
- **Top features.** L1 f1113 (score 38.2, rise 73.2), L5 f93 (specificity 0.89), L1 f564, L7 f280, L0 f1178.
- **Paint vs object.** As in v1, the features cannot tell paint over the bowl from paint over an empty spot. Across the 29 features, Δ(occluded) and Δ(occluded_absent) correlate at r = 0.9975 (median ratio 0.98). π0.5 sees one frame at a time, so these are "gray patch where the *target* is" features, not object permanence.

### 4.5 Goal 2: the features do not drive the action (FAIL, and now a tight null)

All 18 layers patched at once, held-out test frames (34 of 40 after the gap rule), 10 tasks:

| Patch (inject, all tokens) | Gap closed % | 95% CI (task bootstrap) |
|---|---|---|
| **29 occlusion features** (= uncapped = target-specific) | **0.27** | −0.32, 0.92 |
| random, same norm (10 draws) | 0.05 | −0.01, 0.11 |
| random, same norm, uncapped size (3 draws) | 0.03 | −0.10, 0.15 |
| random, raw (2 draws) | 0.05 | −0.11, 0.20 |
| colour features (34) | 0.05 | −0.10, 0.17 |
| all 36,864 transcoder features | 42.77 | 35.94, 50.64 |
| MLP-output swap (ceiling) | 76.61 | 68.51, 84.78 |

- **Occlusion − random = +0.21 pts [−0.35, +0.84].**
  - Sign-flip p = 0.222 over tasks and 0.25 over frames.
  - Positive in 7 of 10 tasks and 65% of frames.
- **Larger but inconsistent per-frame effects.** Per frame, the selected features move the action more than random features do: median |gap closed| 0.75 vs 0.22 points. The sign is inconsistent, though:
  - they beat every one of the 10 random draws on 12 of 34 frames (about 3 would be expected if they were interchangeable with random features);
  - they lose to every draw on 8.
  - The selected features perturb the action more than random directions, but not systematically toward the occluded action.
- **Occlusion − colour = +0.22 [−0.39, +0.86]**, p = 0.19.
- **Verdict: margin +0.21 vs the +10 rule → FAIL.** The upper CI bound of the difference is 12× below the threshold, so this is not a power problem.
- **Share of the ceiling.** The selected features carry 0.35% of the MLP ceiling and 0.62% of what all transcoder features carry. All transcoder features carry 55.8% of the ceiling.
- **Per frame,** the occlusion patch ranges from −3.8% to +7.1%. Per-task differences from random range −1.1 to +2.0 pts (tasks 0–2 negative).

### 4.6 Robustness of the null

| Subset | Frames / tasks | Occlusion % [CI] | Occlusion − random [CI] |
|---|---|---|---|
| all included | 34 / 10 | 0.27 [−0.32, 0.92] | +0.21 [−0.35, +0.84] |
| without the 3 included test frames from failed rollouts | 31 / 9 | 0.19 [−0.46, 0.91] | +0.15 [−0.48, +0.83] |
| without the largest-gap frame (1.79) | 33 / 10 | 0.27 [−0.33, 0.96] | +0.22 [−0.36, +0.87] |
| rollout tasks 1–9 only | 30 / 9 | 0.44 [−0.15, 1.08] | +0.38 [−0.18, +0.98] |
| gap ≥ 0.2 only | 18 / 9 | 0.61 [−0.24, 1.74] | +0.56 [−0.26, +1.68] |
| task 0 only (demo frames; v1's task) | 4 / 1 | −1.02 | −1.03 |

No subset comes within a factor of 5 of the +10-point bar. This table does not undo the other uses of failed rollouts:
- 2 discovery frames (task 8) came from a failed rollout and shaped selection;
- 3 failed rollouts supplied extra training frames;
- task 4's test rollouts never lifted the bowl, so their frames sit at t = 114 and 223;
- the target is invisible in the wrist camera in 5 test frames (26–29, 35).

### 4.7 How distributed is the effect?

k-curve, gap closed % (all layers, held-out frames):

| K features | 50 | 100 | 200 | 500 | 1,000 | 2,000 | 5,000 | all 36,864 |
|---|---|---|---|---|---|---|---|---|
| top-K by this frame's activation change | 0.02 | 0.12 | 1.04 | 6.71 | 11.91 | 18.18 | 29.80 | 42.77 |
| top-K by discovery-frame change | – | 0.03 | – | 2.07 | – | 6.47 | 10.45 | |
| random K | – | – | – | 0.61 | – | 3.84 | – | |

- **Magnitude-ranked sets need thousands of features.** The best per-frame magnitude ranking needs 5,000 features (13.6% of the dictionary) to reach 29.8%, which is 70% of the all-feature effect. It beats random K by 11× at K = 500 and 4.7× at K = 2,000.
- **Rankings transfer only partly.** The discovery ranking transfers weakly (1.7× random at K = 2,000). Its top 500 overlap the per-frame top 500 by 60%.
- **Magnitude ≠ causal relevance.**
  - Half of each frame's occlusion-induced activation change (y-scale-weighted L1) sits in a median of 35.5 features, 80% in 306 and 90% in 1,446. Yet the top 50 close only 0.02%.
  - The reason is where the change lives: 79.8% of it is in **layers 16–17** (73.1% in L17 alone).
    - All top-50 discovery-ranked features are in L16–17.
    - For the per-frame ranking, a proxy computed from the saved test-frame sums (|Σ_t Δf|) puts a median 100% (min 94%) of the top 50 in L16–17.
    - There the MLP output's RMS norm is 1,608 (L16) and 7,697 (L17), against 36–71 in layers 1–10.
  - **L17 is causally inert.** Its output never reaches the expert: residual swap 0.0%, splice 0.00 of the gap.
  - **L16 is nearly inert.** Residual swap 0.39%, MLP swap 0.17%, splice 0.08 of the gap.
- **Where the selected features rank.** The 29 carry 0.04% of the activation change. Their median rank is in the top 13% of features, and none is among the per-frame top 29.

### 4.8 Held-out replication and target specificity

- **Replication.** Of the 29 selected features, **8 (27.6%) pass the full Goal-1 rule again on the 40 test frames**: L1 f22, L4 f397, L5 f93, L5 f1232, L5 f1992, L7 f1272, L10 f1285, L15 f730. The other 21 do not.
  - 0% of the random draws and 0% of the colour set pass it.
  - Only 12 of 36,802 live features pass on test frames (0.033%), so 8 of the 12 are selected features: **846× enrichment**.
- **Response correlation.** Across all live features, the per-feature occlusion response on discovery and test frames correlates at Spearman ρ = 0.65 (Pearson 0.80).
- **On the test frames,** the selected features have median specificity 0.66, are positive on a median 88% of frames, and have a median distractor/target response ratio of 0.058. Two features are not target-specific on test frames (L0 f1178 0.51, L2 f2045 0.44).
- **Conclusion.** The selection picks up a real, partly reproducible, largely target-selective signal. The replication is across demonstrations within the same 10 tasks, not across tasks. What fails is sufficiency, not existence.

### 4.9 Scopes, remove direction, single layers, single features, edges (all on held-out frames)

- **Token scope.**
  - Occlusion features: bowl tokens only 0.26%, other tokens 0.09%.
  - MLP ceiling: bowl tokens 26.7% vs other tokens 61.4%. Most of the MLP-mediated occlusion computation happens off the painted patches.
- **Remove direction** (occluded run → base):

  | Patch | Gap closed % | 95% CI |
  |---|---|---|
  | occlusion features | 0.51 | −1.95, 3.51 |
  | colour | −0.06 | |
  | MLP swap | 88.0 | 86.5, 89.3 |

  MLP computation is necessary for the effect; the selected features are not.
- **Single Goal-1 layers:**

  | Layer | Occlusion % | Random % | Ceiling % |
  |---|---|---|---|
  | L5 | −0.03 | 0.10 | 12.5 |
  | L1 | −0.07 | 0.01 | 13.4 |
  | L7 | 0.11 | 0.04 | 12.7 |

  All three margins are < 0.2 pts.
- **Single features:** L5 f93 0.12% [0.03, 0.21], L1 f564 0.09% [0.04, 0.16], L1 f1113 0.03% [−0.07, 0.13]. Two of the three are reliably above zero, but tiny.
- **Edges.** Injecting the L5 set turns on 2.4% of the L7 set's occlusion response (random: 0.3%). L1→L5 is 0.6% and L1→L7 0.3%. There is a weak serial chain, the same size as v1's L4→L5 (2.3%).

### 4.10 Goal 3: closed loop (exploratory)

Task 0, 10 episodes per condition, the 29 features at 13 layers:

| Condition | Success | Median steps (successes) |
|---|---|---|
| clean reference | 0.9 | 76 |
| bowl painted in both cameras ("hidden") | 0.9 | 78 |
| hidden + features ablated (×0) | 0.8 | 78 |
| hidden + features amplified (×3) | 0.8 | 77.5 |

- **Check 6 passes:** baseline success is strictly between 0 and 1.
- **Neither intervention shifts success.** Ablated and amplified are each −0.10 [−0.30, 0.00] vs hidden, from a single discordant episode (init 1); McNemar p = 1.0.
- **A likely reason "hidden" does not hurt.** In the recorded episode (episode 0, agentview) the paint keeps the bowl's silhouette and moves with it, and the robot grasps the gray blob. This manipulation removes the target's appearance but not its shape or location, which matches the feature analysis ("gray patch where the target is"). With n = 10, only large shifts would be detectable.

---

## 5. Interpretation

1. **Existence vs sufficiency.** v2 separates two claims that v1 could not.
   - **Existence: strong.** π0.5 has a small, partly reproducible, largely target-specific set of occlusion features.
   - **Sufficiency: rejected.** Those features carry essentially none of the net occlusion effect on the action, with a CI that excludes the pre-specified effect by more than 10×. They do perturb the action more than random features, but not consistently toward the occluded action.
2. **The occlusion signal is a distributed change in many token representations.**
   - The full residual swap works by layer 1–7.
   - One MLP carries ≤ 22%; all MLPs together 77%; all 36,864 dictionary features 43%.
   - The best 5,000 features by activation change reach 30%.
   - We found no small, concept-aligned subset of per-layer MLP transcoder features that carries it. Attribution-based rankings remain untested.
3. **Magnitude-based rankings mislead in VLAs.** Layers 16–17 dominate activation-change rankings (massive activations), yet they barely influence the action. The expert reads per-layer K/V, so the last layer's outputs go nowhere and L16's reach it only through L17's K/V. Attribution-based rankings restricted to layers ≤ 13 should be used instead.
4. **Compare runs relative to their controls.** From v1 to v2, the selected set's absolute effect fell 14.7×, but the random and colour controls fell 10.5× and 14.4×.
   - v2 patches a tenth of the dictionary share v1 did (0.08% vs 0.78%), so absolute effects are not comparable across the runs.
   - What changed is the inference. v1's 6.9× ratio over random (p = 0.125, 4 selected frames) looked like above-chance causality. On held-out, unselected frames the ratio is 4.9× but the paired difference is indistinguishable from zero.
   - v1's design (frames selected for the largest effect, features selected and tested on the same frames) is a known source of inflation. These data cannot separate that from the other changes.
5. **Behaviour agrees.** A silhouette-preserving occluder does not reduce success, and the occlusion features do not change it.

---

## 6. Limitations and threats to validity

- **Mixed frame sources.** Only task 0 had demonstrations, so tasks 1–9 used policy rollouts. 7 of 45 rollouts failed.
  - 8 probe frames come from failed rollouts. Three of them were patched in Goal 2, and removing those three does not change the result (§4.6).
  - The other five do not count toward Goal 2. Three were excluded by the gap rule, and two (task 8) are discovery frames that did shape selection.
  - Extra training frames from 3 failed rollouts also entered the transcoders.
- **Transcoder fidelity out of sample.** Test-frame FVU is 0.40, and mid-layer splices distort the test-frame action by about the size of the gap. The dictionary is trained on about 138 distinct scenes and does not generalise well. Interventions stay exact (the error term is kept), but a better dictionary could represent more of the effect with fewer features. Remedies: more demonstrations per task, cross-layer transcoders, or residual SAEs at layers 1–7.
- **Synthetic occluder.** Flat gray paint keeps the silhouette, and π0.5 conditions on a single frame, so the study cannot test object permanence. `occluded_absent` is nearly pixel-identical to `occluded` by construction.
- **Ranking by magnitude, not attribution.** The magnitude-ranked k-curve only upper-bounds how many features are needed. An attribution-ranked curve could be steeper.
- **Replication scope.** Held-out frames come from different demonstrations (or rollouts) of the *same* 10 tasks, so replication across tasks is untested.
- **Frame-level tests.** Frame-level p-values (e.g. the target-vs-distractor sign test) treat frames as independent. The task-level tests are the conservative ones.
- **One policy, one suite, one object type.** Goal 3 uses one task and 10 episodes per condition. It is labelled exploratory and has low power (it can detect only large shifts).
- **Threshold changes.** Three thresholds were adapted before the v2 data (check 3, check 4, Goal 1 positivity). Under v1's strict rules, check 3 fails on one frame and Goal 1 keeps 14 of the 29 features. The thresholds were pre-specified, not registered.

---

## 7. Suggested next experiments

1. Attribution patching (gradient × activation-difference) to rank features, restricted to layers 0–13, then a k-curve on held-out frames.
2. Download demonstrations for all 10 tasks and retrain the transcoders on about 10× more scenes; check the test-frame FVU.
3. Residual-stream SAEs at layers 1–7, where a single swap closes ≥ 93%.
4. A physical occluder (a rendered box in front of the bowl) and a policy with history, for a real permanence contrast.
5. Goal 3 with 50 episodes per condition across 3–5 tasks, plus an "absent" closed-loop condition, which moves the action 2.8× more than paint.

---

## 8. Files in this repository

| Path | What it is |
|---|---|
| `RESULTS.md` | auto-generated one-page summary written by the Colab download cell |
| `REPORT.md` | this report |
| `paper/` | ICLR 2027 paper: `main.tex`, `main.pdf`, `references.bib`, figures, style files |
| `analysis/derive_stats.py` → `derived_stats.json` | recomputes every quoted number from `results/` |
| `analysis/make_paper_figures.py` | builds `paper/figures/fig_{protocol,layers,goal2,heldout}` and the appendix figures |
| `analysis/check_claims.py` | checks that every quoted number appears, correctly formatted, in the report and paper |
| `analysis/v1_derived_stats.json` | verbatim v1 numbers (smaj-research @ 5fc0c90) used for the comparison |
| `code/tc_occlusion.py` | the whole pipeline (7 subcommands, chainable in one process) |
| `code/config.json`, `code/notebook.ipynb` | exact configuration and the Colab notebook with all outputs |
| `results/run_v2.log` | full log of the run |
| `results/status/*.json` | per-step pass/fail and summaries; `timings.json` |
| `results/validate.json` | checks 1–4, per-frame action RMSE under every condition, per-layer swap profiles |
| `results/train.json` | per-layer transcoder metrics (held-out and test-frame FVU, splice, training curves) |
| `results/goal1_feature_table.csv` | all 36,864 features × every Goal-1 statistic |
| `results/goal1_selected.json`, `goal1_rank.pt` | selected, colour, target-specific and random-pool feature sets; per-feature discovery statistics |
| `results/goal2_rows.csv` | every one of the Goal-2 patching runs (frame × site × method × scope × direction × draw) |
| `results/goal2_stats.json` | aggregated Goal-2 statistics (CIs, tests, k-curve, replication, edges) |
| `results/goal2_circuit_trace_table.{md,csv}`, `goal2_frames.json`, `goal2_edges.json` | summary table, per-frame gaps and overlap, edges |
| `results/goal2_test_feature_sums.npy` | 40 test frames × 7 conditions × 18 layers × 2,048 feature sums (held-out replication) |
| `results/goal3.json`, `results/videos/*.mp4` | closed-loop per-episode outcomes; first-episode videos per condition |
| `results/frames_candidates.json` | every candidate frame considered (visibility search) |
| `results/fig_*.png` | figures produced by the run |
| `env/` | package versions, GPU |

Reproduce the analysis on a CPU:

```bash
pip install numpy torch matplotlib pillow opencv-python
python analysis/derive_stats.py && python analysis/make_paper_figures.py && python analysis/check_claims.py
cd paper && latexmk -pdf main.tex
```

---

## 9. v1 vs v2: comparison of results

Both runs use the same model (π0.5 LIBERO fine-tune), the same questions, the same gap metric and the same +10-point Goal-2 rule.

| | **v1** (smaj-research) | **v2** (this repo) |
|---|---|---|
| **Design** | | |
| Tasks | 1 (LIBERO-Spatial task 0) | 10 (all of LIBERO-Spatial) |
| Probe frames | 4 | 60 (20 discovery + 40 held-out test) |
| Frame choice | largest base-vs-occluded gap of 8 candidates | fixed phases 0.4/0.8 of the pre-grasp window |
| Selection vs test frames | same 4 frames | disjoint demonstrations |
| Frame source | demonstrations | demonstrations (task 0), policy rollouts (tasks 1–9) |
| Conditions | 6 | 7 (+ `occluded_distractor`) |
| Transcoder training data | 26,852 tokens, 47 passes, includes the test frames | 146,599 tokens, 258 passes, discovery only |
| Transcoders | 512 features, k = 16, 3,000 steps | 2,048 features, k = 32, 4,000 steps (8,000 retry) |
| Random draws / CIs / permutation unit | 5 / none / 4 frames | 10 / task bootstrap / 10 tasks and 34 frames |
| Goal-2 primary site | single layers (all-layers added post hoc) | all 18 layers (fixed in advance) |
| Compute | ≈ 32 min (7 processes) | 123 min (1 process) |
| **Validity** | | |
| Occlusion gap (action RMSE) | 0.52 median (0.52–0.67) | 0.186 median (0.015–1.79) |
| Gap / sampling noise | 2.9–4.4× | 2.1× median |
| Best single-layer residual swap | 91.8% (L3) | 97.1% (L1) |
| Max single-layer MLP swap | 6.5% | 22.1% (L12) |
| Median FVU, held-out tokens | 0.174 | 0.185 |
| Median FVU, held-out test frames | n/a (not measured) | 0.400 |
| Action: occlude target vs distractor | n/a | 0.186 vs 0.084; target larger on 44/60 frames |
| **Goal 1** | | |
| Selective features | 72 / 9,216 (0.78%) | 29 / 36,864 (0.08%) |
| Top layers | 5, 0, 4 | 5, 1, 7 |
| Median specificity | 0.64 | 0.64 |
| r(Δocc, Δocc_absent) | 0.9986 | 0.9975 |
| Distractor / target response | n/a | 0.03 (discovery), 0.06 (test) |
| Held-out replication of selectivity | n/a | 8/29 pass on test frames (base rate 0.03%, 846×); ρ = 0.65 |
| **Goal 2 (all 18 layers, inject)** | | |
| Occlusion features | 3.92% | **0.27% [−0.32, 0.92]** |
| Random, same norm | 0.56% | 0.05% [−0.01, 0.11] |
| Colour features | 0.66% | 0.05% [−0.10, 0.17] |
| Occlusion / random | 6.9× | 4.9× (difference not significant) |
| Occlusion − best control (+10 needed) | +3.26 → FAIL | **+0.21 → FAIL** |
| CI of occlusion − random | none | [−0.35, +0.84] |
| Sign-flip p | 0.125 (4 frames) | 0.222 (10 tasks), 0.25 (34 frames) |
| Frames beating / losing to every random draw | 3 / 4 beating | 12 / 34 beating, 8 / 34 losing |
| Share of the dictionary patched | 0.78% (72 of 9,216) | 0.08% (29 of 36,864) |
| Absolute shrink v1→v2 (selected / random / colour) | – | 14.7× / 10.5× / 14.4× |
| All transcoder features | 40.1% | 42.8% [35.9, 50.6] |
| MLP-output swap (ceiling) | 65.5% | 76.6% [68.5, 84.8] |
| Occlusion as share of ceiling | 6.0% | 0.35% |
| Bowl tokens / other tokens (occlusion features) | 3.08 / 0.96 | 0.26 / 0.09 |
| Bowl tokens / other tokens (MLP ceiling) | 20.2 / 29.4 | 26.7 / 61.4 |
| Remove direction: occlusion features / MLP swap | 7.9% / 85.6% | 0.51% / 88.0% |
| Best single feature | 1.12% (L5 f124) | 0.12% (L5 f93) |
| Strongest edge (occlusion vs random) | L4→L5 2.3% vs 0.04% | L5→L7 2.4% vs 0.3% |
| k-curve (features needed) | not measured | 500: 6.7%, 2,000: 18.2%, 5,000: 29.8% |
| **Goal 3** | not run (gate) | exploratory: clean 0.9, hidden 0.9, ablated 0.8, amplified 0.8 (n.s.) |
| **Headline** | "selective but not sufficient; causal above chance (suggestive)" | "selective, partly replicable, largely target-specific; negligible and inconsistent net causal effect; the effect is distributed" |

**Reading the comparison.**
- **What held up:**
  - the validity battery (it improved);
  - the existence of sparse selective features, now shown to replicate and to be target-specific;
  - the all-feature and ceiling effects (40→43%, 66→77%);
  - a weak serial edge between two Goal-1 layers (2.3% and 2.4%).
- **What did not hold up:** v1's reading that the selected features act above chance.
  - The ratio over random went from 6.9× (p = 0.125, 4 selected frames) to 4.9× with a paired difference of +0.21 [−0.35, +0.84] on held-out, unselected frames.
  - On task 0 itself, the held-out estimate is −1.0% (4 frames).
- **What is not comparable:** the absolute sizes. The selected set's 3.92% → 0.27% (14.7×) is matched by the controls (random 10.5×, colour 14.4×). v2 patches a ten times smaller share of a four times wider dictionary.
- **Caveat:** v2 changed dictionary, set size, frame choice, split and tasks at once. The comparison cannot attribute any change to a single factor, such as selection bias.
