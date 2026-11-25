"""Re-solve the Experiment 1 conditions that did not close within 300 s, with a 3,600 s limit.

The 300 s results are kept as results/exp1_uncert_<condition>.json and the re-solved results
replace results/exp1/<condition>.json."""
import json, os, shutil
import run_exp1

if __name__ == '__main__':
    for fn in sorted(os.listdir(run_exp1.OUT)):
        path = os.path.join(run_exp1.OUT, fn)
        r = json.load(open(path))
        if r['status'] != 'Optimal':
            shutil.copy(path, os.path.join('results', 'exp1_uncert_' + fn))
            os.remove(path)
            run_exp1.run((r['family'], r['rho'], r['lam'], r['par']), time_limit=3600)
