"""Real learners: Duolingo half-life regression data (Settles & Meeder 2016), user-level sample.
Self-model of a word = competence left at its last practice, calibrated from the learner's past accuracy b
(x = a + c b, which absorbs regression to the mean), propagated with an assumed forgetting rate dhat.
Tests Proposition 1 (gap of a frozen self-model grows with staleness as (x-phi)(1-e^{-lam t}))
and Theorem 2 (false mastery under stability bias; none in expectation when dhat >= lam)."""
import numpy as np, pandas as pd, json
from scipy.optimize import least_squares
d = pd.read_csv('../data/duolingo_sample.csv.gz')
d = d[d.history_seen >= 3].copy()
d['days'] = d.delta / 86400.0
d['b'] = d.history_correct / d.history_seen
d['w'] = d.session_seen.astype(float)
users = d.user_id.unique()
rng = np.random.default_rng(2027)
fitu = set(rng.choice(users, len(users) // 2, replace=False))
fit_df, te = d[d.user_id.isin(fitu)], d[~d.user_id.isin(fitu)].copy()
def model(p, b, t, lam=None):
    a, c, phi, l = p
    l = l if lam is None else lam
    x = a + c * b
    return phi + (x - phi) * np.exp(-l * t)
r = least_squares(lambda p: np.sqrt(fit_df.w.values) * (model(p, fit_df.b.values, fit_df.days.values) - fit_df.p_recall.values),
                  x0=[0.3, 0.65, 0.8, 0.05], bounds=([-1, 0, 0, 0], [1, 2, 1, 5]))
a, c, phi, lam = r.x
print('fit half: x = %.3f + %.3f b, phi = %.3f, lambda = %.4f/day (half-life of the excess %.1f days)' % (a, c, phi, lam, np.log(2) / lam))
TAU = 0.92
bins = [0, 0.01, 0.1, 1, 3, 7, 14, 30, 90, 1000]
labels = ['<15 min', '15 min-2.4 h', '2.4 h-1 d', '1-3 d', '3-7 d', '7-14 d', '14-30 d', '30-90 d', '>90 d']
te['bin'] = pd.cut(te.days, bins, labels=False)
def summarise(df):
    out = {}
    x = a + c * df.b.values
    out['pred_gap'] = df.assign(g=(x - phi) * (1 - np.exp(-lam * df.days.values))).groupby('bin').apply(
        lambda z: np.average(z.g, weights=z.w), include_groups=False).reindex(range(9)).tolist()
    for name, dh in [('frozen', 0.0), ('fitted', lam), ('double', 2 * lam)]:
        bel = phi + (x - phi) * np.exp(-dh * df.days.values)
        z = df.assign(bel=bel, err=bel - df.p_recall.values)
        gap = z.groupby('bin').apply(lambda q: np.average(q.err, weights=q.w), include_groups=False).reindex(range(9))
        cert = z[z.bel >= TAU]
        rc = cert.groupby('bin').apply(lambda q: np.average(q.p_recall, weights=q.w), include_groups=False).reindex(range(9))
        out[name] = dict(gap=gap.tolist(), recall_cert=rc.tolist(), share_cert=float(len(cert) / len(z)),
                         recall_cert_all=float(np.average(cert.p_recall, weights=cert.w)) if len(cert) else np.nan,
                         overconf_cert=float(np.average(cert.bel - cert.p_recall, weights=cert.w)) if len(cert) else np.nan,
                         false_cert=float(cert.w[cert.bin.isin(BAD)].sum() / cert.w.sum()) if len(cert) else 0.0)
    return out
x_hi = a + c * 1.0
K = np.log((TAU - phi) / (x_hi - phi)) / -lam
BAD = [k for k in range(9) if bins[k] >= K]     # delay bins entirely beyond the predicted crossing time
res = summarise(te)
grp = {u: q for u, q in te.groupby('user_id')}; tu = list(grp)
boots = [summarise(pd.concat([grp[u] for u in rng.choice(tu, len(tu))], ignore_index=True)) for _ in range(200)]
def ci(f):
    A = np.array([f(b_) for b_ in boots], float)
    return np.nanpercentile(A, 2.5, axis=0).tolist(), np.nanpercentile(A, 97.5, axis=0).tolist()
res['pred_gap_ci'] = ci(lambda b_: b_['pred_gap'])
for k in ['frozen', 'fitted', 'double']:
    for s_ in ['gap', 'recall_cert', 'share_cert', 'recall_cert_all', 'overconf_cert', 'false_cert']:
        res[k][s_ + '_ci'] = ci(lambda b_: b_[k][s_])
res.update(a=a, c=c, phi=phi, lam=lam, tau=TAU, x_max=x_hi, K_days=K, bad_bins=BAD, labels=labels,
           n_rows=int(len(d)), n_users=int(len(users)), n_pairs=int(d.groupby(['user_id', 'lexeme_id']).ngroups),
           n_test=int(len(te)), n_per_bin=te.groupby('bin').size().reindex(range(9)).tolist(),
           span_days=float((d.timestamp.max() - d.timestamp.min()) / 86400))
json.dump(res, open('../results/duolingo.json', 'w'), indent=1, default=float)
print('crossing time K for a word left at x=%.3f: %.1f days; tau=%.2f' % (x_hi, K, TAU))
print('bins', labels); print('n', res['n_per_bin'])
print('predicted gap', np.round(res['pred_gap'], 3))
for k in ['frozen', 'fitted', 'double']:
    R = res[k]
    print(k, 'gap', np.round(R['gap'], 3))
    print('   recall of certified', np.round(R['recall_cert'], 3))
    print('   share certified %.3f, certified beyond K %.3f %s, overconfidence of certified %.3f %s' % (R['share_cert'], R['false_cert'], np.round(R['false_cert_ci'], 3), R['overconf_cert'], np.round(R['overconf_cert_ci'], 3)))
