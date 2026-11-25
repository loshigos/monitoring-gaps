"""Experiment 2: reward-trained recurrent and transformer agents."""
import json, os, sys, time, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
T = 25
ERR = 1.5
OUT = 'results/exp2'
os.makedirs(OUT, exist_ok=True)

# token ids: 0 start, 1-3 cue E/H/X, 4 blank, 5 '+', 6 '-'
NTOK = 7


def sample(fam, par, rho, n, rng):
    """Returns tokens (n, T+1), y (n,), irr (n,) and ideal-observer confidence (n, T+1)."""
    y = rng.integers(0, 2, n)  # 0 = '+'
    if fam == 'A':
        irr = rng.random(n) < rho
        u = rng.random((n, T)); u2 = rng.random((n, T))
        beta = 0.3
        bb = np.where(irr, beta + par, beta)[:, None]
        blank = u < bb
        corr_sym = np.where(u2 < 0.65, y[:, None], 1 - y[:, None])
        rand_sym = (u2 < 0.5).astype(int)
        sym = np.where(irr[:, None], rand_sym, corr_sym)
        obs = np.where(blank, 4, 5 + sym)
        tok = np.concatenate([np.zeros((n, 1), int), obs], 1)
        # posterior over (tau, y): log-likelihoods
        lik = {}
        nb = np.cumsum(obs == 4, 1); npl = np.cumsum(obs == 5, 1); nmi = np.cumsum(obs == 6, 1)
        z = np.zeros((n, 1))
        nb = np.concatenate([z, nb], 1); npl = np.concatenate([z, npl], 1); nmi = np.concatenate([z, nmi], 1)
        pc, pw = 0.7 * 0.65, 0.7 * 0.35
        bI = beta + par; sI = (1 - bI) / 2
        lR0 = nb * np.log(beta) + npl * np.log(pc) + nmi * np.log(pw)
        lR1 = nb * np.log(beta) + npl * np.log(pw) + nmi * np.log(pc)
        lI = nb * np.log(bI) + (npl + nmi) * np.log(max(sI, 1e-300)) if sI > 0 else np.where(npl + nmi > 0, -np.inf, nb * np.log(bI))
        L = np.stack([np.log(1 - rho) + lR0, np.log(1 - rho) + lR1, np.log(rho) + lI, np.log(rho) + lI], -1)
        L = L - L.max(-1, keepdims=True)
        P = np.exp(L); P /= P.sum(-1, keepdims=True)
        pplus = P[..., 0] + P[..., 2]
    else:  # C
        tp = rng.choice(3, n, p=[(1 - rho) / 2, (1 - rho) / 2, rho])  # 0 easy 1 hard 2 hopeless
        cue = np.where(rng.random(n) < par, tp, rng.integers(0, 3, n))
        qs = np.array([0.75, 0.6, 0.5])[tp]
        u = rng.random((n, T))
        sym = np.where(u < qs[:, None], y[:, None], 1 - y[:, None])
        obs = 5 + sym
        tok = np.concatenate([(1 + cue)[:, None], obs], 1)
        irr = tp == 2
        z = np.zeros((n, 1))
        npl = np.concatenate([z, np.cumsum(obs == 5, 1)], 1); nmi = np.concatenate([z, np.cumsum(obs == 6, 1)], 1)
        prior_t = np.array([(1 - rho) / 2, (1 - rho) / 2, rho])
        Ls = []
        for k, qk in enumerate((0.75, 0.6, 0.5)):
            cl = np.where(cue == k, par + (1 - par) / 3, (1 - par) / 3)[:, None]
            base = np.log(prior_t[k]) + np.log(cl)
            Ls.append(base + npl * np.log(qk) + nmi * np.log(1 - qk))   # y = '+'
            Ls.append(base + npl * np.log(1 - qk) + nmi * np.log(qk))   # y = '-'
        L = np.stack(Ls, -1)
        L = L - L.max(-1, keepdims=True)
        P = np.exp(L); P /= P.sum(-1, keepdims=True)
        pplus = P[..., 0] + P[..., 2] + P[..., 4]
    conf = np.maximum(pplus, 1 - pplus)
    return tok, y, irr, conf


class GRUAgent(nn.Module):
    def __init__(self, h=64, e=32):
        super().__init__()
        self.emb = nn.Embedding(NTOK, e)
        self.rnn = nn.GRU(e, h, batch_first=True)
        self.pi = nn.Linear(h, 4); self.v = nn.Linear(h, 1)

    def forward(self, tok):
        x, _ = self.rnn(self.emb(tok))
        return self.pi(x), self.v(x).squeeze(-1)


class TransAgent(nn.Module):
    def __init__(self, d=64, heads=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(NTOK, d)
        self.pos = nn.Embedding(T + 1, d)
        lay = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers)
        self.pi = nn.Linear(d, 4); self.v = nn.Linear(d, 1)
        self.register_buffer('mask', torch.triu(torch.full((T + 1, T + 1), float('-inf')), 1))

    def forward(self, tok):
        L = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(L))
        x = self.enc(x, mask=self.mask[:L, :L], is_causal=True)
        return self.pi(x), self.v(x).squeeze(-1)


def rollout(logits, y, lam, greedy=False, gen=None):
    """Actions over the exogenous stream. 0 continue, 1 answer '+', 2 answer '-', 3 abstain.
    Returns stop time tau, terminal action, step log-probs and entropies."""
    n = logits.shape[0]
    lg = logits.clone()
    lg[:, T, 0] = -1e9  # must stop at T
    logp = F.log_softmax(lg, -1)
    if greedy:
        a = lg.argmax(-1)
    else:
        a = torch.distributions.Categorical(logits=lg).sample()
    cont = (a == 0)
    # tau = first index where action != continue
    notc = (~cont).int()
    tau = torch.argmax(notc, 1)  # first stop (exists since forced at T)
    term = a[torch.arange(n), tau]
    yt = torch.as_tensor(y)
    rew = torch.where(term == 3, 0.0, torch.where(term - 1 == yt, 1.0, -ERR))
    return a, tau, term, rew, logp


def train(fam, par, arch, seed, rho=0.4, lam=None, updates=2000, batch=512):
    lam = (0.02 if fam == 'A' else 0.06) if lam is None else lam
    fn = f'{OUT}/{fam}_{par}_{arch}_{seed}.json'
    if os.path.exists(fn):
        return json.load(open(fn))
    torch.manual_seed(seed); rng = np.random.default_rng(seed + 12345)
    net = GRUAgent() if arch == 'gru' else TransAgent()
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    t0 = time.time()
    for u in range(updates):
        frac = min(1.0, u / (0.4 * updates))
        lam_u, rho_u = lam * frac, rho * frac
        ent_c = 0.05 + (0.002 - 0.05) * u / updates
        tok, y, irr, conf = sample(fam, par, max(rho_u, 1e-6), batch, rng)
        logits, val = net(torch.as_tensor(tok))
        a, tau, term, rew, logp = rollout(logits, y, lam_u)
        steps = torch.arange(T + 1)[None, :]
        active = (steps <= tau[:, None]).float()
        # return-to-go from step t: -lam*(tau - t) + rew
        G = (-lam_u * (tau[:, None] - steps).float() + rew[:, None]) * active
        lp = logp.gather(-1, a[..., None]).squeeze(-1)
        adv = (G - val).detach()
        ent = -(logp.exp() * logp).sum(-1)
        loss = -(adv * lp * active).sum() / active.sum() + 0.5 * (((G - val) ** 2) * active).sum() / active.sum() \
               - ent_c * (ent * active).sum() / active.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    # evaluation
    net.eval()
    rng_e = np.random.default_rng(999)
    rets, rec = [], []
    with torch.no_grad():
        for b in range(30):
            tok, y, irr, conf = sample(fam, par, rho, 2000, rng_e)
            logits, _ = net(torch.as_tensor(tok))
            a, tau, term, rew, _ = rollout(logits, y, lam, greedy=True)
            ret = rew.numpy() - lam * tau.numpy()
            rets.append(ret)
            rec.append((tau.numpy(), irr, conf, term.numpy()))
    rets = np.concatenate(rets)
    tau = np.concatenate([r[0] for r in rec]); irr = np.concatenate([r[1] for r in rec])
    conf = np.concatenate([r[2] for r in rec]); term = np.concatenate([r[3] for r in rec])
    dstop, ci = matched_stop(tau, irr, conf)
    out = dict(fam=fam, par=par, arch=arch, seed=seed, ret=float(rets.mean()), se=float(rets.std() / np.sqrt(len(rets))),
               dstop=dstop, dstop_ci=ci, mean_tau=float(tau.mean()), abstain=float((term == 3).mean()),
               abstain_t0=float(((term == 3) & (tau == 0)).mean()), train_time=time.time() - t0)
    json.dump(out, open(fn, 'w'))
    print(out, flush=True)
    return out


def cells(tau, irr, conf, nbins=50):
    """Per (t, bin): counts of running and stopping items by type."""
    n = len(tau)
    stats = {}
    for t in range(T):  # decisions at t<T
        run = tau >= t
        if not run.any():
            break
        b = np.minimum(np.floor((conf[run, t] - 0.5) / 0.01 + 1e-9).astype(int), nbins - 1)
        st = (tau[run] == t)
        ii = irr[run]
        for k in np.unique(b):
            m = b == k
            nI = int((m & ii).sum()); nR = int((m & ~ii).sum())
            sI = int((m & ii & st).sum()); sR = int((m & ~ii & st).sum())
            stats[(t, int(k))] = (nI, sI, nR, sR)
    return stats


def dstop_from_stats(stats, minc=20):
    num = den = 0.0
    for (t, k), (nI, sI, nR, sR) in stats.items():
        if nI >= minc and nR >= minc:
            w = min(nI, nR)
            num += w * (sI / nI - sR / nR); den += w
    return num / den if den > 0 else float('nan')


def matched_stop(tau, irr, conf, boot=200):
    st = cells(tau, irr, conf)
    d = dstop_from_stats(st)
    rng = np.random.default_rng(0)
    bs = []
    n = len(tau)
    for b in range(boot):
        idx = rng.integers(0, n, n)
        bs.append(dstop_from_stats(cells(tau[idx], irr[idx], conf[idx])))
    bs = np.array(bs)
    return float(d), [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


if __name__ == '__main__':
    jobs = []
    for arch in ('gru', 'transformer'):
        for seed in range(3):
            for par in (0.0, 0.2, 0.4, 0.6):
                jobs.append(('A', par, arch, seed))
            for par in (0.0, 0.3, 0.6, 1.0):
                jobs.append(('C', par, arch, seed))
    arch_sel, part, nparts = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    jobs = [j for j in jobs if j[2] == arch_sel]
    for j in jobs[part::nparts]:
        train(*j)
