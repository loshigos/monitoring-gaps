# monitoring-gaps

Code and results for the article

> Caro MF, Toscano RE. *When is confidence enough for metacognitive control? Certified monitoring gaps in artificial agents.* Submitted to *Cognitive Computation*.

The repository computes, for evidence-accumulation tasks, the **monitoring gap**: the value lost by the best controller that reads only a monitor (such as confidence) relative to the optimal controller. The best monitor-restricted controller is computed exactly with a mixed-integer linear programme (HiGHS), so every gap is certified rather than estimated. The repository also trains recurrent and transformer agents from reward and compares their stopping behaviour with the certified gaps.

All numbers, tables and figures in the article and in Online Resource 1 can be regenerated from the files in `results/`, and every file in `results/` can be regenerated from the code.

## Installation

Python 3.10 or later.

```bash
pip install -r requirements.txt
```

Everything runs on CPU; no GPU is required.

## Repository layout

| File | Content |
|---|---|
| `core.py` | Task families A–C, exact dynamic programming, monitors, information quantities, local search and the certification programme |
| `gauss.py` | Family D (Gaussian evidence) on a discretised belief grid |
| `multi.py` | Multi-action acquisition problem (Experiment 3) and its certification programme |
| `agents.py`, `agents_gauss.py` | Reward-trained agents (GRU and transformer for families A and C; GRU and LSTM for family D) |
| `results/` | All computed results (JSON; trained family D agents as NPZ) |
| `figures/` | All figures, in PNG and PDF |
| `tables/` | Tables 2 and 3 of the article (Markdown) |

## Reproducing the results

Run the scripts from the repository root, in this order. Approximate run times are for a single CPU core.

| Step | Command | Output | Time |
|---|---|---|---|
| 1. Validate the certification programme against exhaustive enumeration | `python validate_milp.py` | `results/milp_validation.json` | minutes |
| 2. Experiment 1 (256 conditions, families A–C) | `python run_exp1.py <n_processes>` | `results/exp1/` | hours (median 2 s per condition; a few take up to 300 s) |
| 3. Re-solve the conditions that did not close within 300 s | `python resolve_exp1_uncertain.py` | `results/exp1/`, `results/exp1_uncert_*.json` | up to 1 h per condition |
| 4. Recompute the projected-controller analysis | `python postprocess_exp1_projected.py` | `results/exp1/` | minutes |
| 5. Summary of Experiment 1 | `python analyse_exp1.py` | `results/summary_exp1.json` | seconds |
| 6. Monitor refinements (progress and solvability monitors) | `python run_exp1b.py <n_processes>` | `results/exp1b/` | hours |
| 7. Certified references for Experiment 2 | `python run_exp2_refs.py` | `results/exp2_refs/` | minutes |
| 8. Experiment 2 agents (48) | `python agents.py gru 0 1` and `python agents.py transformer 0 1` | `results/exp2/` | 2–6 min per agent |
| 9. Summary of Experiment 2 | `python analyse_exp2.py` | `results/summary_exp2.json` | seconds |
| 10. Experiment 3 (54 configurations, 108 programmes) | `python run_exp3.py` | `results/exp3/` | under 1 h |
| 11. Summary and local-optimality check of Experiment 3 | `python analyse_exp3.py` and `python check_multi_local.py` | `results/summary_exp3.json`, `results/multi_local_check.json` | minutes |
| 12. Family D phase diagram | `python run_gauss.py` | `results/gauss/` | hours (200 s limit per cell) |
| 13. Decision information for family D | `python postprocess_gauss_dinfo.py` | `results/gauss/` | minutes |
| 14. Family D agents (90) | `python agents_gauss.py 0 1` | `results/gauss_agents/` | several minutes per agent |
| 15. Probing and steering of family D agents | `python probe_steer.py` | `results/probe_steer.json` | minutes |
| 16. Summary and figures of family D | `python analyse_gauss.py` | `results/summary_gauss.json`, `results/gauss_agents_summary.json`, Fig. 6, Fig. S2 | minutes |
| 17. Remaining figures and tables | `python fig_schematic.py`, `python figs.py fig2 fig3 fig4 figS4`, `python figs_refinements_agents_tables.py` | Figs. 1–5, 7, S4; Tables 2–3 | seconds |

The agent scripts take a part index and a number of parts (for example `python agents_gauss.py 0 4`, ..., `python agents_gauss.py 3 4`), so that training can be split across processes or machines. Random seeds are fixed inside the scripts.

## Figures and tables

| Article item | File | Produced by |
|---|---|---|
| Fig. 1 | `figures/fig1_schematic` | `fig_schematic.py` |
| Fig. 2 | `figures/fig2_certified` | `figs.py fig2` |
| Fig. 3 | `figures/fig3_predictors` | `figs.py fig3` |
| Fig. 4 | `figures/fig4_bounds` | `figs.py fig4` |
| Fig. 5 | `figures/fig5_refinements` | `figs_refinements_agents_tables.py` |
| Fig. 6 (and Fig. S1) | `figures/fig6_gauss` | `analyse_gauss.py` |
| Fig. 7 | `figures/fig7_agents` | `figs_refinements_agents_tables.py` |
| Fig. S2 | `figures/figS2_gauss_agents` | `analyse_gauss.py` |
| Fig. S4 | `figures/figS4_sources` | `figs.py figS4` |
| Table 2 | `tables/table2.md` | `figs_refinements_agents_tables.py` |
| Table 3 | `tables/table3.md` | `figs_refinements_agents_tables.py` |

## License

MIT (see `LICENSE`).

## Citation

See `CITATION.cff`. The article reference will be added on publication.
