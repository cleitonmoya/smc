# -*- coding: utf-8 -*-
"""
Sequential Importance Sampling
Modelo Self Avoid Walk (SAW) utilizado para (bio) polímeros
Baseado em: 
    http://www.stat.ucla.edu/~zhou/courses/Stats102C-IS.pdf
Implementação nossa.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

# Verifica quais vizinhos estão livres
def getVizinhosLivres(x, celulasOcupadas):
    i,j = x
    vizinhos = [(i,j-1), (i,j+1), (i-1,j), (i+1,j)]
    vizinhosOcupados = set(celulasOcupadas).intersection(vizinhos)
    vizinhosLivres = list(set(vizinhos)-vizinhosOcupados)
    return vizinhosLivres

# Obtém uma amostra pelo método Sequential Importance Sampling
def getSampleWeighted(T):
    w = 1
    xt = (0,0)
    celulasOcupadas = [xt]
    
    for _ in range(T-1):
        vizinhosLivres = getVizinhosLivres(xt, celulasOcupadas)
        nt = len(vizinhosLivres)
        if nt > 0:
            xt = tuple(rng.choice(vizinhosLivres))
            celulasOcupadas.append(xt)
            w = w*nt
        else:
            w = 0
            break
    return celulasOcupadas,w


# Estima Z_T por força bruta e por SIS
T = np.arange(3,21,dtype=int) 
N = 100000

Z_sis = []
Z_bf = []

for j,t in enumerate(T):
    print(f"Estimando Zt para T={t}")
    Caminhos = []
    W = []
    for n in range(N):
        c,w = getSampleWeighted(t)
        if w != 0:
            W.append(w)
            if t < 11:
                if not c in Caminhos:
                    Caminhos.append(c)
    z_sis = np.mean(W).round().astype(int)
    Z_sis.append(z_sis)
    print(f"\tpor SIS: {z_sis}")
    
    # Realiza também estimativa por força-bruta para t<11
    if t < 11:
        z_bf = len(Caminhos)
        Z_bf.append(z_bf)
        print(f"\tpor força bruta: {z_bf}")
        

# Plota os resultados
fig, ax = plt.subplots(layout='constrained', figsize=(5,2.5))
ax.plot(T, Z_sis, 'o-', label='SIS')
ax.scatter(np.arange(3,3+len(Z_bf)), Z_bf, marker='x', color='red', label='força-bruta', zorder=2)
ax.set_xticks(T)
ax.set_yscale('log')
ax.set_ylabel(r'$\hat{Z}_T$')
ax.set_xlabel('T')
ax.grid()
ax.legend()