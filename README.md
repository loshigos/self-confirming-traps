# Self-confirming traps in metacognitive self-regulation

Code and results for the article *Self-confirming traps in metacognitive self-regulation* (M.F. Caro, submitted to Artificial Intelligence).

Everything runs on a standard CPU in minutes; no GPU, paid API or large model is needed.

## Installation
```
pip install -r requirements.txt
cd code
```
Run every script from the `code/` folder.

## Scripts and what they reproduce
| Script | Reproduces |
|---|---|
| `rsr.py` | Reflexive self-regulation model (Definition 1) and the policies of Definition 4; imported by the other scripts |
| `exp_model.py` | Table 1 (policies and metamemory) and Table 2 (false mastery: closed form vs simulation) |
| `exp_capacity_noise.py` | Self-knowledge capacity (Theorem 3) and assessment noise experiments (Figure 3) |
| `exp_bkt.py` | Mastery-learning tutor with Bayesian knowledge tracing and forgetting students (Figure 4) |
| `exp_neural.py` | Neural network learning digit tasks (Table 3) |
| `rerun_mastery.py` | Re-runs the mastery configurations with the stay-until-mastered rule (updates the JSON files) |
| `duolingo_muestra_colab.py` | Draws the Duolingo learner sample (see `data/README.md`) |
| `exp_duolingo.py` | Language-learner analysis, Section 7.7 (needs `data/duolingo_sample.csv.gz`) |
| `figs.py` | Figures 2, 3 and 4 |
| `fig_concept.py` | Figure 1 (conceptual figure) |
| `fig_duolingo.py` | Figure 5 |

Recommended order: `exp_model.py`, `exp_capacity_noise.py`, `exp_bkt.py`, `exp_neural.py`, `rerun_mastery.py`, `exp_duolingo.py`, then the three figure scripts.

## Folders
- `results/`: JSON outputs and logs used for the tables and figures in the article.
- `figures/`: figures as submitted (`Figure_1.pdf` to `Figure_5.pdf`). The scripts write `fig_concept.pdf` (Figure 1), `fig1.pdf` (Figure 2), `fig2.pdf` (Figure 3), `fig3.pdf` (Figure 4) and `fig_duolingo.pdf` (Figure 5) to the same folder.
- `data/`: location for the Duolingo sample, which is not redistributed.

## Data
Duolingo spaced-repetition data, Harvard Dataverse, https://doi.org/10.7910/DVN/N8XJME (Settles & Meeder, 2016).

## License
Code under the MIT License (see `LICENSE`). The Duolingo data keep the terms of their original release.
