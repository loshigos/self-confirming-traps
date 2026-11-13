"""Simulated students in a mastery-learning tutor with Bayesian knowledge tracing (BKT).
True students forget unpractised skills; the tutor's student model assumes forget rate pF_hat."""
import numpy as np, json
pL0, pT, g, s = 0.2, 0.2, 0.2, 0.1
def simulate(policy, pF, pF_hat, n=10, T=800, N=500, K=None, seed=0, tau=0.95):
    rng = np.random.default_rng(seed)
    know = rng.random((N, n)) < pL0
    P = np.full((N, n), pL0)
    stale = np.zeros((N, n), int)
    cur = np.full(N, -1)
    curve = []
    for t in range(T):
        if policy == 'mastery':
            # stay on the current skill until P(L) >= tau, then the unmastered skill with lowest P(L)
            m = np.where(P < tau, P, np.inf)
            nxt = np.argmin(m, axis=1); idle = ~np.isfinite(m.min(axis=1))
            keep = (cur >= 0) & (P[np.arange(N), np.maximum(cur, 0)] < tau)
            a = np.where(keep, cur, nxt); idle = idle & ~keep
            cur = np.where(idle, -1, a)
        elif policy == 'remedial':
            a = np.argmin(P, axis=1); idle = np.zeros(N, bool)
        if K is not None:
            over = stale.max(axis=1) >= K
            a = np.where(over, np.argmax(stale, axis=1), a); idle = idle & ~over
        rows = np.arange(N)
        k = know[rows, a]
        correct = np.where(k, rng.random(N) > s, rng.random(N) < g)
        p = P[rows, a]
        post = np.where(correct, p*(1-s)/(p*(1-s)+(1-p)*g), p*s/(p*s+(1-p)*(1-g)))
        post = post + (1-post)*pT
        # forgetting for unpractised skills (true and modelled)
        forget = rng.random((N, n)) < pF
        prac = np.zeros((N, n), bool); prac[rows[~idle], a[~idle]] = True
        know = know & ~(forget & ~prac)
        P = np.where(prac, P, P*(1-pF_hat))
        learn = rng.random(N) < pT
        newk = k | learn
        know[rows[~idle], a[~idle]] = newk[~idle]
        P[rows[~idle], a[~idle]] = post[~idle]
        stale += 1; stale[prac] = 0
        fm = ((P >= tau) & ~know).sum() / max(1, (P >= tau).sum())
        curve.append((float(fm), float(know.mean()), float(np.abs(P - know).mean()), float((~idle).mean())))
    return curve
out = {}
for name, pol, pFh, K in [('mastery_BKT', 'mastery', 0.0, None), ('mastery_BKT_forget', 'mastery', 0.01, None),
                          ('mastery_BKT_review', 'mastery', 0.0, 10), ('remedial_BKT', 'remedial', 0.0, None),
                          ('remedial_BKT_forget', 'remedial', 0.01, None)]:
    c = simulate(pol, pF=0.01, pF_hat=pFh, K=K)
    out[name] = c
    print(name, 'false-mastery %.3f known %.3f abs.err %.3f active %.2f' % tuple(np.mean(c[-200:], axis=0)))
json.dump(out, open('../results/bkt.json', 'w'))
