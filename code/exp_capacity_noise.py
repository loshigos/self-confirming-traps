import numpy as np, json
from rsr import run
phi, eta, dmax = 0.1, 0.2, 0.02
# ---------- E3 capacity: round robin + midpoint propagation vs adversarial forgetting
def rr_midpoint(n, T=3000, seed=0):
    rng = np.random.default_rng(seed)
    th = np.full(n, 0.5); x = th.copy(); stale = np.zeros(n, int); worst = 0
    dcur = rng.choice([0.0, dmax], n)
    for t in range(T):
        bel = phi + (x - phi) * (1 + (1 - dmax) ** stale) / 2
        if t > 50 * n: worst = max(worst, np.max(np.abs(bel - th)))
        a = t % n
        # adversary: each gap uses a constant rate 0 or dmax, redrawn at each practice
        new = phi + (1 - dcur) * (th - phi)
        new[a] = th[a] + eta * (1 - th[a]); x[a] = new[a]; dcur[a] = rng.choice([0.0, dmax])
        th = new; stale += 1; stale[a] = 0
    return worst
E3 = []
for n in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
    w = max(rr_midpoint(n, seed=s) for s in range(10))
    up = (1 - phi) / 2 * (1 - (1 - dmax) ** (n - 1))
    E3.append(dict(n=n, worst_sim=float(w), minimax=float(up)))
    print(E3[-1])
def capacity(e):
    return 1 + int(np.floor(np.log(1 - 2 * e / (1 - phi)) / np.log(1 - dmax)))
E3cap = {str(e): capacity(e) for e in [0.01, 0.02, 0.05, 0.1]}
print('capacity', E3cap)
# ---------- E4 assessment noise
sigmas = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
E4 = {}
for pol in ['greedy', 'remedial', 'mastery']:
    for n in [2, 4]:
        for sg in sigmas:
            e, c = [], []
            for s in range(20):
                r = run(pol, n=n, eta=eta, delta=0.02, delta_hat=0.0, phi=phi, sigma=sg, T=20000, rng=np.random.default_rng(100 + s))
                e.append(r['err'][-10000:].mean()); c.append(r['comp'][-10000:].mean())
            E4[f'{pol}|{n}|{sg}'] = (float(np.mean(e)), float(np.std(e) / np.sqrt(20)), float(np.mean(c)), float(np.std(c) / np.sqrt(20)))
            print(pol, n, sg, np.round(E4[f'{pol}|{n}|{sg}'], 3))
json.dump(dict(E3=E3, E3cap=E3cap, E4=E4), open('../results/model_E3_E4.json', 'w'), indent=1)
