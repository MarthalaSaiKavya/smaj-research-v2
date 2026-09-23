# smaj-research-v2: held-out transcoder circuit analysis of occlusion in π0.5

**Question.** When the object named in its instruction is hidden, how does the vision-language-action policy π0.5 register it, and does that representation drive its actions? We apply per-layer TopK transcoders and causal patching to π0.5's 18-layer Gemma backbone on all 10 LIBERO-Spatial tasks.

**What v2 changes.** Protocol v2 re-runs the v1 study ([smaj-research](https://github.com/MarthalaSaiKavya/smaj-research)) with:
- features selected on discovery demonstrations and tested on held-out ones;
- frames chosen at fixed phases, not by effect size;
- an identical-distractor control;
- task-clustered confidence intervals;
- a k-curve of effect vs number of features;
- an exploratory closed-loop test.

**Result.**
- **The features are real.** 29 target-specific occlusion features exist, and their selectivity partly replicates on held-out frames: 8 of the 29 pass again, 846× over the base rate, Spearman ρ = 0.65.
- **Their net effect on the action is negligible.** Together they close **0.27% [−0.32, 0.92]** of the occlusion effect on the action, against 0.05% for random features. That is far below the pre-specified +10-point bar, with a CI that excludes it.
- **The effect is distributed.** All transcoder features close 42.8% and the MLP outputs 76.6%.
- **v1's apparent above-chance effect is not confirmed.** v1 found 6.9× random on 4 selected frames. Absolute effects are not comparable across the runs, because the controls shrank almost as much as the selected set.

| Start here | |
|---|---|
| [`REPORT.md`](REPORT.md) | Full research report: problem, method, novelty, every result, limitations, **v1 vs v2 comparison table** (§9) |
| [`paper/main.pdf`](paper/main.pdf) | ICLR 2027 submission draft (anonymous), *Selective but Causally Negligible: Held-Out Transcoder Analysis of Occlusion in a Vision-Language-Action Policy*, built from [`paper/main.tex`](paper/main.tex) |
| [`RESULTS.md`](RESULTS.md) | Auto-generated one-page summary from the Colab run |

## Layout
- `code/`
  - `tc_occlusion.py`: the whole pipeline (`frames`, `validate`, `capture`, `train`, `goal1`, `goal2`, `goal3`, chainable in one process);
  - `config.json`;
  - `notebook.ipynb`: the Colab notebook with all outputs.
- `results/`: raw outputs of the run, including the log, status files, per-feature tables, every patching run, held-out feature sums, figures and closed-loop videos.
- `analysis/`
  - `derive_stats.py`: recomputes every quoted number into `derived_stats.json`;
  - `make_paper_figures.py`: builds the paper figures;
  - `check_claims.py`: verifies the numbers quoted in the report and paper against the data;
  - `v1_derived_stats.json`: the v1 numbers used for the comparison.
- `paper/`: LaTeX source, bibliography, figures and the compiled PDF (ICLR 2027 style).
- `env/`: exact package versions and GPU info.

## Reproduce the analysis (CPU is enough)
```bash
pip install numpy torch matplotlib pillow opencv-python
python analysis/derive_stats.py        # -> analysis/derived_stats.json
python analysis/make_paper_figures.py  # -> paper/figures/
python analysis/check_claims.py        # 144 quoted numbers vs. the data
cd paper && latexmk -pdf main.tex      # -> paper/main.pdf
```
Re-running the experiments needs a GPU with lerobot 0.6.1 and LIBERO. See `code/notebook.ipynb`: one cell runs all seven steps in about 2 h on an A100.
