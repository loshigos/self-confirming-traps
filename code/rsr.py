"""Reflexive self-regulation (RSR) model: learning on practice, forgetting otherwise,
self-model updated only on the practised region; agent's own forgetting model delta_hat."""
import numpy as np

def clip(v, lo, hi): return np.minimum(np.maximum(v, lo), hi)

def run(policy, n=4, eta=0.2, delta=0.02, delta_hat=0.0, phi=0.1, sigma=0.0,
        T=4000, theta0=None, tau=0.9, K=None, rng=None, delta_seq=None):
    rng = np.random.default_rng(0) if rng is None else rng
    th = np.array(theta0, float) if theta0 is not None else rng.uniform(0.3, 0.7, n)
    n = len(th)
    bh = th.copy()                 # exact initial self-model
    stale = np.zeros(n, int)
    err, comp, acts = np.zeros(T), np.zeros(T), np.zeros(T, int)
    lp = np.full(n, np.inf)        # stored learning progress
    cur = -1
    for t in range(T):
        err[t] = np.max(np.abs(bh - th)); comp[t] = th.mean()
        # --- choose
        if K is not None and stale.max() >= K:
            a = int(np.argmax(stale))
        elif policy == 'greedy':
            m = bh.max(); c = np.flatnonzero(bh >= m - 1e-12); a = int(c[np.argmax(stale[c])])
        elif policy == 'remedial':
            m = bh.min(); c = np.flatnonzero(bh <= m + 1e-12); a = int(c[np.argmax(stale[c])])
        elif policy == 'mastery':
            # mastery learning: stay on the current skill until its estimate reaches tau,
            # then move to the unmastered skill with the lowest estimate; idle if none
            if cur >= 0 and bh[cur] < tau: a = cur
            else:
                c = np.flatnonzero(bh < tau)
                a = -1 if len(c) == 0 else int(c[np.argmin(bh[c])])
            cur = a
        elif policy == 'lp':
            m = lp.max(); c = np.flatnonzero(lp >= m - 1e-12); a = int(c[np.argmax(stale[c])])
        elif policy == 'roundrobin':
            a = int(np.argmax(stale))
        acts[t] = a
        d = delta if delta_seq is None else delta_seq(t, n, rng)
        d = np.broadcast_to(d, (n,)).astype(float)
        # --- observe + belief update
        newb = phi + (1 - delta_hat) * (bh - phi)
        if a >= 0:
            # noisy self-assessment of the post-practice competence (noise enters through the observation y)
            post = th[a] + eta * (1 - th[a])
            newb[a] = clip(post + (1 - eta) * sigma * rng.standard_normal(), phi, 1.0)
            lp[a] = newb[a] - bh[a] if np.isfinite(lp[a]) else newb[a] - bh[a]
            lp[a] = abs(newb[a] - bh[a])
        # --- truth update
        newth = phi + (1 - d) * (th - phi)
        if a >= 0: newth[a] = th[a] + eta * (1 - th[a])
        th, bh = newth, newb
        stale += 1
        if a >= 0: stale[a] = 0
    return dict(err=err, comp=comp, acts=acts)

if __name__ == '__main__':
    for pol in ['greedy', 'remedial', 'mastery', 'lp', 'roundrobin']:
        r = run(pol, T=20000)
        gaps = []
        for i in range(4):
            idx = np.flatnonzero(r['acts'] == i)
            gaps.append(np.diff(idx).max() if len(idx) > 1 else -1)
        print(pol, 'err_end %.3f err_avg %.3f comp %.3f maxgap' % (r['err'][-5000:].max(), r['err'][-5000:].mean(), r['comp'][-5000:].mean()), gaps,
              'gaps late', [np.diff(np.flatnonzero(r['acts'][10000:]==i)).max() if (r['acts'][10000:]==i).sum()>1 else -1 for i in range(4)])
