##################################################################################################
### vibes: Variance-bounded state-spaces

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import scipy as sc
from scipy.spatial import ConvexHull

import jax
import jax.numpy as jp
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

##################################################################################################

delta = lambda i, j: 1 if i == j else 0

def construct_P(n, r, alpha):
    D = np.diag([1] + (r-1)*[1/alpha])
    @jax.jit
    def obj(V):
        O = jp.hstack([jp.ones((n,1))/jp.sqrt(n), V.reshape(n,r-1)])
        return jp.linalg.norm(O.T @ O - jp.eye(r)) +\
               jp.linalg.norm(jp.diag(O[:, 1:] @ O[:,1:].T) - jp.ones(n)*(r-1)/n) +\
               (jp.sum(O @ D @ O.T) - jp.sum(abs(O @ D @ O.T)))**2
    result = sc.optimize.minimize(obj, np.random.randn(n*(r-1)),\
                                jac=jax.jit(jax.jacrev(obj)),\
                                tol=1e-16, options={"disp": True, "maxiter": 10000})
    O = jp.hstack([jp.ones((n,1))/jp.sqrt(n), result.x.reshape(n,r-1)])
    P = abs(np.array(O @ D @ O.T))
    P = P/np.sum(P, axis=0)
    return P
    
def extract_P_data(P):
    alphas = [l for l in np.linalg.eigvals(P) if not np.isclose(l, 0) and not np.isclose(l,1)]
    n = P.shape[0]
    r = len(alphas)+1
    alpha = 1/alphas[0]
    beta = (1-alpha)/n
    gamma = n/(1 + (r-1)/alpha)
    Phi = alpha*np.eye(n) + beta*np.ones((n,n))
    return {"n":n, "r": r, "alpha": float(alpha), "beta": float(beta),\
            "gamma": float(gamma), "Phi": Phi}

def check_P(P):
    P_data = extract_P_data(P)
    n, gamma, Phi = P_data["n"], P_data["gamma"], P_data["Phi"]
    tests = {"non-negative": bool(np.all(P >= 0)),\
             "stochastic": np.allclose(np.sum(P, axis=0), np.ones(n)),\
             "symmetric": np.allclose(P, P.T),\
             "unbiased": np.allclose(np.diag(P), np.ones(n)/gamma),\
             "depolarizing": np.allclose(P @ Phi @ P, P)}
    return tests 

def C_constructor(P, A=None):
    P_data = extract_P_data(P)
    n, alpha, beta, Phi = P_data["n"], P_data["alpha"], P_data["beta"], P_data["Phi"]
    if type(A) == type(None):
        eta = -beta/(alpha+1)
        A = np.array([[[eta*(delta(i,j) - delta(i,k) - delta(j,k)) for k in range(n)] for j in range(n)] for i in range(n)])
    V = np.array([[[delta(i,j)*delta(i,k) for k in range(n)] for j in range(n)] for i in range(n)])
    @jax.jit
    def C(x):
        return P @ Phi @ jp.einsum("ijk, k", V-A, x) @ P @ Phi
    return C

def valid_probabilities(m, P, A=None):
    C = C_constructor(P, A=A)
    count = 0
    valid = []
    while len(valid) != m:
        p = np.random.dirichlet(np.ones(P.shape[0]))
        if np.all(np.linalg.eigvals(C(p)) >= 0):
            valid.append(p)
        count += 1
    return np.array(valid), len(valid)/count

##################################################################################################

def embedding_operator(n):
    simplex0 = np.eye(n) - np.ones(n)/n
    L, D, R = np.linalg.svd(simplex0)
    F = R[:-1,:]
    p = np.random.dirichlet(np.ones(n))
    if np.allclose(p, F.T @ (F @ (p - np.ones(n)/n)) + np.ones(n)/n):
        return F

def viz(P, m=3000, A=None, title="3in3", embed=False):
    P_data = extract_P_data(P)
    n, alpha, r = P_data["n"], P_data["alpha"], P_data["r"]

    if not embed:
        simplex = np.eye(n)
        valid, ratio = valid_probabilities(m, P, A=A)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot() if n == 2 else fig.add_subplot(projection="3d")
        ax.set_title("%s | alpha=%.3f | r=%d | %.3f%%" % (title, alpha, r, 100*ratio))
        ax.azim = 25
        ax.scatter(*simplex, c="r")
        ax.scatter(*P, c="g", marker="x", s=500)
        ax.scatter(*valid.T, alpha=0.04)
    else:
        F = embedding_operator(n)
        tiny_simplex = (F @ (np.eye(n) - np.ones((n,n))/n)).T
        tiny_P = (F @ (P - np.ones((n,n))/n)).T
        valid, ratio = valid_probabilities(m, P)
        tiny_valid = (F @ (valid.T - np.ones((n, m))/n)).T
        hull = ConvexHull(tiny_valid)
        hull_pts = tiny_valid[hull.vertices]

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot() if n == 3 else fig.add_subplot(projection="3d")
        ax.set_title("%s | alpha=%.3f | r=%d | %.3f%%" % (title, alpha, r, 100*ratio))
        ax.azim = 25
        ax.scatter(*tiny_simplex.T, c="r")
        ax.scatter(*tiny_P.T, c="g", marker="x", s=750)
        ax.scatter(*tiny_valid.T, alpha=0.04)
        ax.scatter(*hull_pts.T, alpha=0.3, c="y")

    plt.savefig("%s_a%.3f_r%d_v%.3f.png" % (title, alpha, r, 100*ratio))
    plt.show()

##################################################################################################

def minimize_unbiased_frame_potential(d, n, t):
    min_fp = 1/sc.special.binom(d+t-1, t)
    @jax.jit
    def frame_potential(V):
        R = V[:d*n].reshape(d, n) + 1j*V[d*n:].reshape(d, n)
        R = R/jp.linalg.norm(R, axis=0)
        return ((1/n**2)*jp.sum(abs(R.conj().T @ R)**(2*t)) - min_fp)**2
    
    result = sc.optimize.minimize(frame_potential, np.random.randn(2*d*n),\
                                  jac=jax.jit(jax.jacrev(frame_potential)),\
                                  tol=1e-26, options={"disp": True, "maxiter": 10000})
    V = result.x
    R = V[:d*n].reshape(d, n) + 1j*V[d*n:].reshape(d, n)
    R = R/np.linalg.norm(R, axis=0)
    S = [np.outer(r, r.conj()) for r in R.T]
    E = [(d/n)*e for e in S] # 3-design POVM elements
    P = np.array([[(e@s).trace() for s in S] for e in E]).real
    Phi = (d+1)*np.eye(n) - (d/n)*np.ones((n,n))
    return {"d": d, "n": n, "t": t, "P": P, "Phi": Phi, "E": E, "S": S}
