"""Neural self-regulation on sklearn digits: 5 two-class tasks, shared MLP, interference-driven forgetting.
The agent's self-model of task accuracy is updated only from the minibatch it practises (m samples)."""
import numpy as np, json
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
tasks = [(2*i, 2*i+1) for i in range(5)]
tr = [np.flatnonzero(np.isin(ytr, c)) for c in tasks]; te = [np.flatnonzero(np.isin(yte, c)) for c in tasks]
def init(rng, h=32):
    return [rng.normal(0, 0.3, (64, h)), np.zeros(h), rng.normal(0, 0.3, (h, 10)), np.zeros(10)]
def fwd(W, x):
    hdn = np.maximum(0, x @ W[0] + W[1]); return hdn, hdn @ W[2] + W[3]
def acc(W, idx, Xs, ys, c):
    _, z = fwd(W, Xs[idx]); zz = z[:, list(c)]; pred = np.array(c)[zz.argmax(1)]
    return float((pred == ys[idx]).mean())
def step(W, idx, lr=0.05):
    x, t = Xtr[idx], ytr[idx]; hdn, z = fwd(W, x)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    p[np.arange(len(t)), t] -= 1; p /= len(t)
    gW2 = hdn.T @ p; gb2 = p.sum(0); dh = (p @ W[2].T) * (hdn > 0)
    W[0] -= lr * x.T @ dh; W[1] -= lr * dh.sum(0); W[2] -= lr * gW2; W[3] -= lr * gb2
def run(policy, m=32, dh=0.0, T=1500, tau=0.9, seed=0, K=None):
    rng = np.random.default_rng(seed); W = init(rng)
    n = 4; bel = np.full(n, 0.5); stale = np.zeros(n, int); err, accs = [], []; cur = -1
    for t in range(T):
        true = np.array([acc(W, te[j], Xte, yte, tasks[j]) for j in range(n)])
        err.append(np.abs(bel - true).mean()); accs.append(true.mean())
        if K is not None and stale.max() >= K: a = int(np.argmax(stale))
        elif policy == 'greedy':
            c = np.flatnonzero(bel >= bel.max() - 1e-12); a = int(c[np.argmax(stale[c])])
        elif policy == 'remedial':
            c = np.flatnonzero(bel <= bel.min() + 1e-12); a = int(c[np.argmax(stale[c])])
        elif policy == 'mastery':
            if cur >= 0 and bel[cur] < tau: a = cur
            else:
                c = np.flatnonzero(bel < tau); a = 4 if len(c) == 0 else int(c[np.argmin(bel[c])])
            cur = a if a < n else -1
        elif policy == 'roundrobin': a = int(np.argmax(stale))
        bel = 0.5 + (1 - dh) * (bel - 0.5)          # agent's forgetting model (decay toward chance)
        idx = rng.choice(tr[a], 32)
        if a < n:
            ai = rng.choice(tr[a], m)                 # self-assessment sample of size m
            bel[a] = acc(W, ai, Xtr, ytr, tasks[a])
        step(W, idx)
        stale += 1
        if a < n: stale[a] = 0
    return np.array(err), np.array(accs)
if __name__ == '__main__':
    out = {}
    conf = [('mastery', 32, 0.0), ('mastery', 32, 0.01), ('remedial', 32, 0.0), ('greedy', 32, 0.0), ('roundrobin', 32, 0.0)]
    for pol, m, d in conf:
        E, A = [], []
        for s in range(5):
            e, a = run(pol, m=m, dh=d, seed=s); E.append(e[-500:].mean()); A.append(a[-500:].mean())
        out[f'{pol}|m{m}|dh{d}'] = (float(np.mean(E)), float(np.std(E)), float(np.mean(A)), float(np.std(A)))
        print(pol, m, d, np.round(out[f'{pol}|m{m}|dh{d}'], 3), flush=True)
    for m in [1, 2, 4, 8, 32, 128, 300]:
        E, A = [], []
        for s in range(5):
            e, a = run('greedy', m=m, seed=s); E.append(e[-500:].mean()); A.append(a[-500:].mean())
        out[f'greedy_noise|m{m}'] = (float(np.mean(E)), float(np.std(E)), float(np.mean(A)), float(np.std(A)))
        print('greedy m', m, np.round(out[f'greedy_noise|m{m}'], 3), flush=True)
    json.dump(out, open('../results/neural.json', 'w'), indent=1)
