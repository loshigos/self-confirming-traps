import numpy as np, json, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rsr import run
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
C = ['#1b4f72', '#c0392b', '#d68910', '#1e8449', '#7d3c98']
# Figure 1
fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
phi, tau, eta, delta = 0.1, 0.6, 0.2, 0.02
r = run('mastery', n=1, eta=eta, delta=delta, delta_hat=0.0, phi=phi, tau=tau, T=300, theta0=[0.98])
ax[0].plot(r['comp'], color='k', lw=1.4, label='true competence')
for i, dh in enumerate([0.0, 0.005, 0.01]):
    r = run('mastery', n=1, eta=eta, delta=delta, delta_hat=dh, phi=phi, tau=tau, T=300, theta0=[0.98])
    ax[0].plot(r['comp'] + r['err'], color=C[i], lw=1, ls='--', label=r'belief, $\hat\delta=%g$' % dh)
ax[0].axhline(tau, color='grey', lw=0.6, ls=':'); ax[0].text(300, tau + 0.01, r'$\tau$', ha='right')
ax[0].set_xlabel('steps'); ax[0].set_ylabel('competence'); ax[0].set_title('(a) false mastery under stability bias', fontsize=9, loc='left')
ax[0].legend(frameon=False, fontsize=7, loc='lower right', bbox_to_anchor=(1.0, 0.12))
k = np.arange(0, 400)
for i, dh in enumerate([0.0, 0.005, 0.01, 0.015]):
    ax[1].plot(k, (1 - dh) ** k - (1 - delta) ** k, color=C[i], lw=1, label=r'$\hat\delta=%g$' % dh)
ax[1].set_xlabel(r'steps since last practice, $k$'); ax[1].set_ylabel(r'$\Delta(k)$')
ax[1].set_title(r'(b) error profile, $\delta=0.02$', fontsize=9, loc='left'); ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig1.pdf'); plt.close()
# Figure 2
R = json.load(open('../results/model_E3_E4.json'))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
n = [d['n'] for d in R['E3']]
ax[0].plot(n, [d['minimax'] for d in R['E3']], color='k', lw=1, label='minimax value (Thm. 3)')
ax[0].plot(n, [d['worst_sim'] for d in R['E3']], 'o', ms=3, color=C[1], label='round robin, simulated')
ax[0].set_xlabel('regions $n$'); ax[0].set_ylabel('worst-case error'); ax[0].legend(frameon=False, fontsize=7, loc='lower right')
ax[0].set_title('(a) self-knowledge capacity', fontsize=9, loc='left')
sig = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
xs = [1e-3] + sig[1:]
for j, (pol, nn) in enumerate([('greedy', 2), ('greedy', 4), ('remedial', 4)]):
    e = [R['E4'][f'{pol}|{nn}|{s}'][0] for s in sig]; c = [R['E4'][f'{pol}|{nn}|{s}'][2] for s in sig]
    ax[1].plot(xs, e, '-o', ms=2.5, color=C[j], lw=1, label=f'{pol}, n={nn}')
    ax[2].plot(xs, c, '-o', ms=2.5, color=C[j], lw=1, label=f'{pol}, n={nn}')
for a in ax[1:]:
    a.set_xscale('log'); a.set_xlabel(r'noise $\sigma$')
ax[1].set_ylabel('long-run self-model error'); ax[2].set_ylabel('long-run mean competence')
ax[1].set_title('(b) error vs. noise', fontsize=9, loc='left'); ax[2].set_title('(c) competence vs. noise', fontsize=9, loc='left')
ax[1].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig('../figures/fig2.pdf'); plt.close()
# Figure 3: BKT
B = json.load(open('../results/bkt.json'))
lab = {'mastery_BKT': 'mastery, standard BKT', 'mastery_BKT_forget': 'mastery, BKT with forgetting',
       'mastery_BKT_review': 'mastery + review every 10 steps', 'remedial_BKT': 'weakest-first, standard BKT'}
fig, ax = plt.subplots(1, 2, figsize=(7, 2.9))
for i, (k_, l) in enumerate(lab.items()):
    c = np.array(B[k_]); ax[0].plot(c[:, 0], color=C[i], lw=1, label=l); ax[1].plot(c[:, 1], color=C[i], lw=1)
ax[0].set_ylabel('false mastery rate'); ax[1].set_ylabel('fraction of skills known')
for a in ax: a.set_xlabel('tutoring steps')
fig.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False, fontsize=7); ax[0].set_title('(a) mastered but forgotten', fontsize=9, loc='left'); ax[1].set_title('(b) true knowledge', fontsize=9, loc='left')
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig('../figures/fig3.pdf'); plt.close()
print('ok')
