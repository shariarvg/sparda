import numpy as np
import sys, os
from projection import projection
from scipy.stats import wasserstein_distance

def power_iteration(A, num_iter=1000, tol=1e-6):
    n, _ = A.shape
    v = np.random.rand(n)
    v /= np.linalg.norm(v)  # Normalize the vector

    for _ in range(num_iter):
        v_new = np.dot(A, v)
        v_new /= np.linalg.norm(v_new)  # Normalize again
        if np.linalg.norm(v_new - v) < tol:  # Check for convergence
            break
        v = v_new

    return v


def relax(X, Y, T = 10, nu = 0.1, gamma = 0.1, lam = 0.01, n_it=10, verbose = False):
    assert X.shape[1] == Y.shape[1], f"X has feature dimension {X.shape[1]}, Y has {Y.shape[1]}"
    d = X.shape[1]
    n = X.shape[0]
    m = Y.shape[0]
    
    beta = np.ones(d)*np.sqrt(d)/d
    B = np.outer(beta, beta)
    dB = np.zeros((d,d))
    u = np.zeros(n)
    v = np.zeros(m)
    
    best_distance = 0
    last_improvement_step = -1
    best_B = B
    best_beta = np.zeros(B.shape[0])
    counter = 0
    while counter < min(last_improvement_step + T, n_it):
        du = np.ones(n)*1/n
        dv = np.ones(m)*1/m
        
        for i in range(n):
            for j in range(m):
                z = X[i,:] - Y[j,:]
                if np.trace(np.outer(z,z) @ B) - u[i] - v[j] < 0:
                    
                    du[i] = du[i] - 1/m
                    dv[j] = dv[j] - 1/m
                    dB = dB + np.outer(z,z)/m
        
                    
        u = u + nu*du
        v = v + nu*dv
        if verbose:
            print(f"Norm dB: {np.linalg.norm(dB)}")
        
        if np.linalg.norm(dB) != 0:

            increment = gamma/(np.linalg.norm(dB)+1e-8)


            B = projection(B + increment * dB, lam, increment)
        
        ## get objective calculation, possibly update best_distance, last_improvement_step, and best_B
        beta = power_iteration(B)
        
        X_1 = X @ beta
        Y_1 = Y @ beta
        
        obj = wasserstein_distance(X_1, Y_1)
        if verbose:
            print(f"Beta_{counter}: {beta}")
            print(f"Objective_{counter} function: {obj}")
        if obj > best_distance:
            best_distance = obj
            last_improvement_step = i
            best_B = B
            best_beta = beta
            
        counter += 1
        
    return best_B, best_beta