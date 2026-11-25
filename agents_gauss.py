"""GRU (60) and LSTM (30) agents trained from reward on the Gaussian task (family D)."""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from gauss import simulate_items, MU, W, T

torch.set_num_threads(1)
OUT = 'results/gauss_agents'; os.makedirs(OUT, exist_ok=True)
SU = 1.8


class RNNAgent(nn.Module):
    def __init__(self, kind='gru', h=64):
        super().__init__()
        self.rnn = (nn.GRU if kind == 'gru' else nn.LSTM)(2, h, batch_first=True)
        self.pi = nn.Linear(h, 4); self.v = nn.Linear(h, 1)

    def forward(self, inp, return_h=False):
        hs, _ = self.rnn(inp)
        out = (self.pi(hs), self.v(hs).squeeze(-1))
        return out + (hs,) if return_h else out


def make_input(x):
    """x: (n, T) samples -> (n, T+1, 2): [sample (0 at t=0), start flag]."""
    n = x.shape[0]
    inp = np.zeros((n, T + 1, 2), np.float32)
    inp[:, 1:, 0] = x; inp[:, 0, 1] = 1.0
    return torch.as_tensor(inp)


def rollout(logits, y, greedy=False):
    lg = logits.clone(); lg[:, T, 0] = -1e9
    logp = F.log_softmax(lg, -1)
    a = lg.argmax(-1) if greedy else torch.distributions.Categorical(logits=lg).sample()
    tau = torch.argmax((a != 0).int(), 1)
    term = a[torch.arange(len(tau)), tau]
    yt = torch.as_tensor(y)
    # actions: 1 answer +1, 2 answer -1, 3 abstain
    ans = torch.where(term == 1, 1, -1)
    rew = torch.where(term == 3, 0.0, torch.where(ans == yt, 1.0, -W))
    return a, tau, term, rew, logp


def train(kind, lam, rho, seed, updates=2500, batch=512):
    fn = f'{OUT}/{kind}_{lam}_{rho}_{seed}'
    if os.path.exists(fn + '.json'):
        return
    try:
        fd = os.open(fn + '.lock', os.O_CREAT | os.O_EXCL); os.close(fd)
    except FileExistsError:
        return
    torch.manual_seed(seed); rng = np.random.default_rng(1000 + seed)
    net = RNNAgent(kind); opt = torch.optim.Adam(net.parameters(), 3e-3)
    t0 = time.time()
    for u in range(updates):
        rho_u = rho * min(1.0, u / (0.5 * updates))
        ent_c = 0.052 + (0.002 - 0.052) * u / updates
        it = simulate_items(rho_u, SU, batch, rng)
        logits, val = net(make_input(it['x']))
        a, tau, term, rew, logp = rollout(logits, it['y'])
        steps = torch.arange(T + 1)[None, :]
        active = (steps <= tau[:, None]).float()
        Gt = (-lam * (tau[:, None] - steps).float() + rew[:, None]) * active
        lp = logp.gather(-1, a[..., None]).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
        adv = (Gt - val).detach()
        loss = (-(adv * lp * active).sum() + 0.5 * (((Gt - val) ** 2) * active).sum() - ent_c * (ent * active).sum()) / active.sum()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
    net.eval()
    test_seed = int(1e6 + 1000 * lam * 1000 + 10 * rho * 10)
    it = simulate_items(rho, SU, 20000, np.random.default_rng(test_seed))
    with torch.no_grad():
        logits, _ = net(make_input(it['x']))
        a, tau, term, rew, _ = rollout(logits, it['y'], greedy=True)
    ret = rew.numpy() - lam * tau.numpy()
    np.savez_compressed(fn + '.npz', ret=ret, tau=tau.numpy(), term=term.numpy())
    torch.save(net.state_dict(), fn + '.pt')
    out = dict(kind=kind, lam=lam, rho=rho, seed=seed, test_seed=test_seed, ret=float(ret.mean()),
               abstain_all=bool((term.numpy() == 3).all()), mean_tau=float(tau.float().mean()), time=time.time() - t0)
    json.dump(out, open(fn + '.json', 'w'))
    print(out, flush=True)


def jobs():
    J = []
    for lam in (0.0, 0.005, 0.01, 0.02, 0.04):
        for rho in (0.0, 0.2, 0.4, 0.6):
            for s in range(3):
                J.append(('gru', lam, rho, s))
    for lam, rho in ((0.0, 0.4), (0.005, 0.4), (0.01, 0.2), (0.01, 0.4), (0.02, 0.2), (0.02, 0.4), (0.02, 0.6),
                     (0.04, 0.2), (0.04, 0.4), (0.04, 0.6)):
        for s in range(3):
            J.append(('lstm', lam, rho, s))
    return J


if __name__ == '__main__':
    part, nparts = int(sys.argv[1]), int(sys.argv[2])
    J = jobs()
    # high-gap cells first so that the most informative agents finish early
    order = sorted(range(len(J)), key=lambda i: (J[i][1] != 0.02, J[i][2] != 0.4, i))
    for i in order[part::nparts]:
        train(*J[i])
