import numpy as np, json
from rsr import run
R = {}
# E1: policies x metamemory
pols = ['greedy','remedial','lp','mastery','roundrobin']
dhs = [0.0, 0.01, 0.02]   # true delta=0.02
E1 = {}
for p in pols:
    for dh in dhs:
        e, c = [], []
        for s in range(20):
            r = run(p, n=4, delta=0.02, delta_hat=dh, T=20000, rng=np.random.default_rng(s))
            e.append(r['err'][-5000:].mean()); c.append(r['comp'][-5000:].mean())
        E1[f'{p}|{dh}'] = (float(np.mean(e)), float(np.std(e)), float(np.mean(c)), float(np.std(c)))
        print(p, dh, np.round(E1[f'{p}|{dh}'], 3))
R['E1'] = E1
# E2: mastery illusion window and depth: closed form vs simulation
E2 = []
phi, tau, eta = 0.1, 0.6, 0.2
for delta, dh in [(0.02,0.0),(0.02,0.005),(0.02,0.01),(0.05,0.02),(0.01,0.002)]:
    # one region, start just below tau, mastery policy practises until >= tau then idles
    r = run('mastery', n=1, eta=eta, delta=delta, delta_hat=dh, phi=phi, tau=tau, T=6000, theta0=[0.98])
    a = r['acts']; th_err = r['err']
    # reconstruct truth: comp = theta (n=1)
    th = r['comp']; b = th + 0  # need belief; recompute from err sign: belief >= truth under dh<=delta
    bel = th + th_err
    # first idle period
    idle = np.flatnonzero(a == -1); t0 = idle[0]
    x = th[t0]
    kt = np.argmax(th[t0:] < tau); kb = np.argmax(bel[t0:] < tau) if dh > 0 else np.inf
    depth_sim = th[t0 + int(kb)] if dh > 0 else phi
    rho = np.log(1-delta)/np.log(1-dh) if dh > 0 else np.inf
    kt_f = int(np.ceil(np.log((tau-phi)/(x-phi))/np.log(1-delta)))
    kb_f = int(np.ceil(np.log((tau-phi)/(x-phi))/np.log(1-dh))) if dh > 0 else np.inf
    depth_bound = phi + (x-phi)**(1-rho)*(tau-phi)**rho if dh > 0 else phi
    E2.append(dict(delta=delta, dh=dh, x=float(x), kt_sim=int(kt), kt_formula=kt_f, kb_sim=float(kb), kb_formula=float(kb_f),
                   window_sim=float(kb-kt), depth_sim=float(depth_sim), depth_bound=float(depth_bound), rho=float(rho)))
    print(E2[-1])
R['E2'] = E2
json.dump(R, open('../results/model_E1_E2.json','w'), indent=1)
