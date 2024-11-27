# -*- coding: utf-8 -*-
"""
Sequential Importance Sampling
Aplicação: Modelo (não-linear) de Volatilidade Estocástica
Objetivo: 
    - Estimar, de modo online, E[x_t|y_t] através de SIS
    - Verificar o problmema da degeneração dos pesos no SIS
Modelo:
    x[t] = Normal(phi*x[t-1], sigma^2)
    y[t] = Normal(0, exp(gamma + x[t]))
    
    x: volatilidade estocástica (variável latente)
    y: retornos (variável observada)
Baseado em:
    https://www.maths.lancs.ac.uk/~fearnhea/GTP/GTP_Practicals.pdf
Implementação nossa.
"""

import numpy as np
from numpy import sqrt, exp
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8})
rng = np.random.default_rng(seed=42)

# Parâmetros do modelo
phi = 0.95
sigma = sqrt(1-phi**2)
gamma = 1

# Simulação do modelo
T = 100 # número de observações
x_true = np.zeros(T)
y = np.zeros(T)
x_true[0] = rng.normal(loc=0, scale=sigma/sqrt(1-phi**2)) # distribuição estacionária
for t in range(1,T):
    x_true[t] = phi*x_true[t-1] + rng.normal(loc=0, scale=sigma)
    y[t] = rng.normal(loc=0, scale=exp(gamma+x_true[t]))


# Parâmetros do SIS
N = 10 # número de partículas
W = []
X = []
x_est = np.zeros(T) 

# prioris
mu_p = 0
sigma_p = sigma/sqrt(1-phi**2)

# incialização (iteração 0)
x = rng.normal(loc=mu_p, scale=sigma_p, size=N)
X.append(x)
logw = norm.logpdf(y[0], loc=0, scale=exp(gamma + x))

# normalização
w = exp(logw - logsumexp(logw))
W.append(w)

# estimativa de x
x_est[0] = (w*x).sum()

# Laço principal
for t in range(1,T):
    
    # # gera N amostras (partículas) da distribuição proposta q até o istante t
    x = phi*x + rng.normal(loc=0,scale=sigma, size=N)
    X.append(x)
    
    # calcula o log-peso correspondente de cada amostra
    logw = logw + norm.logpdf(y[t], loc=0, scale=exp(gamma + x))
    
    # normalização dos pesos
    w = exp(logw - logsumexp(logw))
    W.append(w)
    
    # estimativa de x
    x_est[t] = (w*x).sum()

X = np.array(X)    
W = np.array(W)


#%% Gráfico da simulação
fig,ax = plt.subplots(nrows=2, layout='constrained', sharex=True, figsize=(6,3))
ax[0].set_title("Retornos observados")
ax[0].plot(y, linewidth=0.5)
ax[0].grid()

ax[1].set_title("Volatilidade estocástica")
ax[1].plot(x_true, linewidth=0.5)
ax[1].grid()
ax[1].set_xlabel('t')

#%% Gráfico com os resultados
fig,ax = plt.subplots(nrows=4, layout='constrained', sharex=True, figsize=(4,4))
ax[0].set_title("Retornos observados")
ax[0].plot(y, label="y")
ax[0].grid()

ax[1].set_title("Volatilidade estocástica")
ax[1].plot(x_est, label=r"$\hat{x}$")
ax[1].plot(x_true, label="x", linestyle='--', color='C2')
ax[1].legend()
ax[1].grid()

ax[2].set_title('Amostras ("partículas")')
ax[2].plot(X, linewidth=0.5)

ax[3].set_title("Pesos normalizados")
ax[3].plot(W, linewidth=0.5)
ax[3].set_xlabel('t')
