import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
R = json.load(open('../results/duolingo.json'))
C = {'frozen': '#c0392b', 'fitted': '#1b4f72', 'double': '#1e8449'}
L = {'frozen': r'frozen self-model ($\hat\delta=0$)', 'fitted': r'fitted forgetting ($\hat\delta=\hat\lambda$)', 'double': r'pessimistic ($\hat\delta=2\hat\lambda$)'}
x = np.arange(9); lab = ['<15m', '15m-2h', '2h-1d', '1-3d', '3-7d', '7-14d', '14-30d', '30-90d', '>90d']
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
lo, hi = R['pred_gap_ci']
ax[0].plot(x, R['pred_gap'], color='k', lw=1, ls='--', label='Proposition 1, fitted on other users')
for k in ['frozen', 'fitted', 'double']:
    g = np.array(R[k]['gap']); l_, h_ = map(np.array, R[k]['gap_ci'])
    ax[0].errorbar(x + {'frozen': -0.12, 'fitted': 0, 'double': 0.12}[k], g, yerr=[g - l_, h_ - g], fmt='o', ms=3, lw=0.8, color=C[k], label=L[k])
ax[0].axhline(0, color='grey', lw=0.5)
ax[0].set_xticks(x, lab, rotation=45, ha='right', fontsize=7); ax[0].set_ylabel('belief minus recall')
ax[0].set_xlabel('time since last practice'); ax[0].set_title('(a) error of the self-model', fontsize=9, loc='left')
ax[0].legend(frameon=False, fontsize=6.5, loc='upper left')
for k in ['frozen', 'fitted']:
    g = np.array(R[k]['recall_cert'], float); l_, h_ = map(lambda v: np.array(v, float), R[k]['recall_cert_ci'])
    ok = ~np.isnan(g)
    ax[1].errorbar(x[ok] + (0.08 if k == 'fitted' else -0.08), g[ok], yerr=[(g - l_)[ok], (h_ - g)[ok]], fmt='o-', ms=3, lw=0.8, color=C[k], label=L[k])
ax[1].axhline(R['tau'], color='grey', ls='-.', lw=0.7); ax[1].text(8.4, R['tau'] + 0.002, r'mastery level $\tau$', ha='right', fontsize=7, color='grey')
kx = np.interp(R['K_days'], [0.005, 0.05, 0.5, 2, 5, 10, 21, 55, 200], x)
ax[1].axvspan(4.5, 8.5, color='#d68910', alpha=0.15, lw=0); ax[1].text(6.5, 0.875, 'beyond predicted\ncrossing time', ha='center', fontsize=7, color='#9a6a00')
ax[1].set_xticks(x, lab, rotation=45, ha='right', fontsize=7); ax[1].set_ylabel('recall of words believed mastered')
ax[1].set_xlabel('time since last practice'); ax[1].set_title('(b) words certified as mastered', fontsize=9, loc='left')
ax[1].legend(frameon=False, fontsize=6.5, loc='lower left'); ax[1].set_ylim(0.86, 0.95)
plt.tight_layout(); plt.savefig('../figures/fig_duolingo.pdf'); plt.savefig('../figures/fig_duolingo.png', dpi=150)
