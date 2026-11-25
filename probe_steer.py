"""Secondary analysis (Appendix B): probing and steering of GRU agents in the Gaussian task (lambda=0.02, rho=0.4)."""
import json, numpy as np, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from gauss import GaussModel, full_grid_voc, interp, simulate_items, T, W
from agents_gauss import RNNAgent, make_input

torch.set_num_threads(1)
LAM, RHO, SU = 0.02, 0.4, 1.8
G = GaussModel(RHO, sigma_u=SU, n_pi=21, n_q=11)
VOCS = full_grid_voc(G, LAM)


def voc_of(items):
    pi = items['pires']; pp = items['pplus']
    q = np.where(pi > 1e-9, (pp - (1 - pi) / 2) / np.maximum(pi, 1e-9), 0.5); q = np.maximum(q, 1 - q)
    return np.stack([interp(VOCS[t], G, pi[:, t], q[:, t]) for t in range(T)], 1)


def hidden(net, items):
    with torch.no_grad():
        _, _, h = net(make_input(items['x']), return_h=True)
    return h.numpy()[:, :T]  # states before decisions t=0..T-1


def gru_rollout_steer(net, items, direction, alpha):
    """Run the GRU step by step, adding alpha*direction to the hidden state after every update."""
    cell = nn.GRUCell(2, 64)
    cell.weight_ih.data = net.rnn.weight_ih_l0.data; cell.weight_hh.data = net.rnn.weight_hh_l0.data
    cell.bias_ih.data = net.rnn.bias_ih_l0.data; cell.bias_hh.data = net.rnn.bias_hh_l0.data
    inp = make_input(items['x']); n = inp.shape[0]
    h = torch.zeros(n, 64); d = torch.as_tensor(direction, dtype=torch.float32)
    alive = torch.ones(n, dtype=torch.bool); tau = torch.full((n,), T); term = torch.zeros(n, dtype=torch.long)
    with torch.no_grad():
        for t in range(T + 1):
            h = cell(inp[:, t], h) + alpha * d
            lg = net.pi(h)
            if t == T:
                lg[:, 0] = -1e9
            a = lg.argmax(-1)
            stop = alive & (a != 0)
            tau[stop] = t; term[stop] = a[stop]
            alive &= (a == 0)
    return tau.float().mean().item(), (term == 3).float().mean().item()


def main():
    rng = np.random.default_rng(123)
    items = simulate_items(RHO, SU, 3000, rng)
    voc = voc_of(items); conf = items['conf'][:, :T]
    bins = np.minimum(np.floor((conf - 0.5) / 0.02 + 1e-9).astype(int), 24)
    # residual VOC: VOC minus its mean within (t, confidence bin)
    res = np.zeros_like(voc)
    for t in range(T):
        for k in np.unique(bins[:, t]):
            m = bins[:, t] == k
            res[m, t] = voc[m, t] - voc[m, t].mean()
    out = {}
    for seed in range(3):
        net = RNNAgent('gru'); net.load_state_dict(torch.load(f'results/gauss_agents/gru_{LAM}_{RHO}_{seed}.pt')); net.eval()
        H = hidden(net, items)
        # use states at t>=1 up to 15,000 samples
        idx = np.array([(i, t) for i in range(H.shape[0]) for t in range(1, T)])
        sel = idx[np.random.default_rng(seed).choice(len(idx), 15000, replace=False)]
        X = H[sel[:, 0], sel[:, 1]]; yres = res[sel[:, 0], sel[:, 1]]; yconf = conf[sel[:, 0], sel[:, 1]]
        alphas = np.logspace(-3, 3, 13)
        r2 = cross_val_score(RidgeCV(alphas=alphas), X, yres, cv=5, scoring='r2').mean()
        r2c = cross_val_score(RidgeCV(alphas=alphas), X, yconf, cv=5, scoring='r2').mean()
        untrained = RNNAgent('gru'); torch.manual_seed(100 + seed); untrained = RNNAgent('gru')
        Hu = hidden(untrained, items); Xu = Hu[sel[:, 0], sel[:, 1]]
        r2u = cross_val_score(RidgeCV(alphas=alphas), Xu, yres, cv=5, scoring='r2').mean()
        # axes
        rv = RidgeCV(alphas=alphas).fit(X, yres).coef_; rv = rv / np.linalg.norm(rv)
        rc = RidgeCV(alphas=alphas).fit(X, yconf).coef_; rc = rc - (rc @ rv) * rv; rc = rc / np.linalg.norm(rc)
        sd_v = (X @ rv).std(); sd_c = (X @ rc).std()
        B = np.stack([rv, rc], 1)
        rnd = []
        g = np.random.default_rng(seed)
        while len(rnd) < 8:
            v = g.normal(size=64); v -= B @ (B.T @ v)
            for u in rnd:
                v -= (u @ v) * u
            v /= np.linalg.norm(v); rnd.append(v)
        sd_r = [(X @ v).std() for v in rnd]
        steer_items = simulate_items(RHO, SU, 6000, np.random.default_rng(999))
        res_steer = {}
        for name, d, sd in [('voc', rv, sd_v), ('conf', rc, sd_c)] + [(f'rand{i}', v, s) for i, (v, s) in enumerate(zip(rnd, sd_r))]:
            res_steer[name] = {str(a): gru_rollout_steer(net, steer_items, d * sd, a) for a in (-4, -2, 0, 2, 4)}
        out[seed] = dict(r2_resvoc=float(r2), r2_resvoc_untrained=float(r2u), r2_conf=float(r2c), steer=res_steer)
        print(seed, out[seed]['r2_resvoc'], out[seed]['r2_resvoc_untrained'], out[seed]['r2_conf'], flush=True)
    json.dump(out, open('results/probe_steer.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
