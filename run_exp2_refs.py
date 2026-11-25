"""Certified references for the Experiment 2 conditions of family A that lie off the Experiment 1 grid
(rho = 0.4, lambda = 0.02, delta in {0.2, 0.4, 0.6}). Output: results/exp2_refs/."""
import os
import run_exp1

run_exp1.OUT = 'results/exp2_refs'
os.makedirs(run_exp1.OUT, exist_ok=True)

if __name__ == '__main__':
    for delta in (0.2, 0.4, 0.6):
        run_exp1.run(('A', 0.4, 0.02, delta))
