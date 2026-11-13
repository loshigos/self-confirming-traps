import numpy as np, json
from rsr import run
from exp_neural import run as nrun
R = json.load(open('../results/model_E3_E4.json'))
for n in [2, 4]:
    for sg in [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        e, c = [], []
        for s in range(20):
            r = run('mastery', n=n, eta=0.2, delta=0.02, delta_hat=0.0, phi=0.1, sigma=sg, T=20000, rng=np.random.default_rng(100 + s))
            e.append(r['err'][-10000:].mean()); c.append(r['comp'][-10000:].mean())
        R['E4'][f'mastery|{n}|{sg}'] = (float(np.mean(e)), float(np.std(e)/np.sqrt(20)), float(np.mean(c)), float(np.std(c)/np.sqrt(20)))
        print('mastery', n, sg, np.round(R['E4'][f'mastery|{n}|{sg}'], 3), flush=True)
json.dump(R, open('../results/model_E3_E4.json', 'w'), indent=1)
N = json.load(open('../results/neural.json'))
for d in [0.0, 0.01]:
    E, A = [], []
    for s in range(5):
        e, a = nrun('mastery', m=32, dh=d, seed=s); E.append(e[-500:].mean()); A.append(a[-500:].mean())
    N[f'mastery|m32|dh{d}'] = (float(np.mean(E)), float(np.std(E)), float(np.mean(A)), float(np.std(A)))
    print('neural mastery', d, np.round(N[f'mastery|m32|dh{d}'], 3), flush=True)
json.dump(N, open('../results/neural.json', 'w'), indent=1)
