"""Conceptual figure: the reflexive self-regulation loop and the self-confirming trap."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'serif', 'font.size': 8.5})
BLUE, RED, GREY, GOLD = '#1b4f72', '#c0392b', '#7f8c8d', '#d68910'
fig = plt.figure(figsize=(7.2, 3.3))
ax = fig.add_axes([0.0, 0.0, 0.53, 1.0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
skills = ['drawers', 'folding', 'pouring', 'wiping']
truth = [0.92, 0.35, 0.55, 0.30]
belief = [0.92, 0.80, 0.70, 0.75]
prac = 0
def box(y, h, title, col):
    ax.add_patch(FancyBboxPatch((0.4, y), 9.2, h, boxstyle='round,pad=0.05,rounding_size=0.25', fc=col, ec='none', alpha=0.10))
    ax.text(0.65, y + h - 0.35, title, fontsize=8.5, fontweight='bold', va='top', color='k')
def bars(y0, vals, filled, hl):
    for i, v in enumerate(vals):
        x = 1.3 + i * 2.1
        ax.add_patch(Rectangle((x, y0), 1.3, 1.9, fc='white', ec=GREY, lw=0.6))
        ec = RED if i == hl else BLUE
        if filled: ax.add_patch(Rectangle((x, y0), 1.3, 1.9 * v, fc=ec, ec='none', alpha=0.85))
        else: ax.add_patch(Rectangle((x, y0), 1.3, 1.9 * v, fc='none', ec=ec, lw=1.3, ls='--'))
        ax.text(x + 0.65, y0 - 0.3, skills[i], ha='center', va='top', fontsize=7.5)
box(6.05, 3.45, 'Meta level: self-model $\\hat\\theta_t$ (what the agent believes)', BLUE)
bars(6.7, belief, False, prac)
box(0.25, 3.75, 'Object level: competence $\\theta_t$ (what is true)', RED)
bars(0.95, truth, True, prac)
# arrows
def arr(p, q, col, txt, tx, ty, rad=0.0, ha='left'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, lw=1.2, color=col, connectionstyle=f'arc3,rad={rad}'))
    ax.text(tx, ty, txt, color=col, fontsize=7.3, ha=ha, va='center')
arr((1.75, 6.2), (1.75, 3.15), RED, '', 0, 0)
ax.text(1.6, 4.7, 'policy $\\pi$:\npractise\none skill', color=RED, fontsize=7.3, ha='right', va='center')
arr((2.35, 3.15), (2.35, 6.2), BLUE, '', 0, 0)
ax.text(2.5, 4.7, 'observe it\n(only it)', color=BLUE, fontsize=7.3, ha='left', va='center')
ax.text(6.6, 5.15, 'practised skill learns ($\\eta$)', color='k', fontsize=7.3, ha='center')
ax.text(6.6, 4.75, 'others forget at true rate $\\delta$', color='k', fontsize=7.3, ha='center')
ax.text(6.6, 4.35, 'beliefs of others decay at $\\hat\\delta$', color='k', fontsize=7.3, ha='center')
ax.text(0.2, 9.95, '(a) reflexive self-regulation', fontsize=9, fontweight='bold', va='top')
# panel b
bx = fig.add_axes([0.62, 0.40, 0.36, 0.50])
phi, x, d = 0.1, 0.95, 0.02; tau = 0.8
k = np.arange(0, 160)
tr = phi + (x - phi) * (1 - d) ** k
for dh, ls, lab in [(0.0, '--', '$\\hat\\delta=0$ (stability bias)'), (0.01, ':', '$\\hat\\delta=0.01$')]:
    bx.plot(k, phi + (x - phi) * (1 - dh) ** k, color=BLUE, ls=ls, lw=1.3, label='belief, ' + lab)
bx.plot(k, tr, color=RED, lw=1.6, label='true competence')
bx.axhline(tau, color=GREY, lw=0.7, ls='-.'); bx.text(158, tau + 0.02, 'mastery level $\\tau$', ha='right', fontsize=7, color=GREY)
kt = np.argmax(tr < tau)
bx.fill_between(k[kt:], tr[kt:], tau, color=GOLD, alpha=0.25, lw=0)
bx.text(105, 0.55, 'false mastery:\nbelieved $\\geq\\tau$,\ntruly below', fontsize=7, color='#9a6a00', ha='center')
bx.set_xlabel('steps without practice'); bx.set_ylabel('competence'); bx.set_ylim(0.05, 1.02)
bx.spines[['top', 'right']].set_visible(False)
bx.legend(frameon=False, fontsize=6.6, loc='lower left')
bx.set_title('(b) the self-confirming trap', fontsize=9, fontweight='bold', loc='left')
fig.text(0.62, 0.20, 'Remedies proved in the paper', fontsize=7.8, fontweight='bold')
fig.text(0.62, 0.035, '• assume forgetting at least as fast as true ($\\hat\\delta\\geq\\delta$)\n'
         '• bound revisit times independently of beliefs\n'
         '• noisy self-assessment escapes exploitation traps', fontsize=7.3, va='bottom', linespacing=1.45)
plt.savefig('../figures/fig_concept.pdf'); plt.savefig('../figures/fig_concept.png', dpi=160)
