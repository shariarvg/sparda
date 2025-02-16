import numpy as np
from qpsolvers import solve_qp
import time

def projection(B, lam, d, verbose = False):
    
    ## Because B is initialized to be 0 and only given outer-product (symmetric) updates, it is guaranteed to be Hermitian
    eigvals, eigvecs = np.linalg.eigh(B)
    
    start = time.time()
    w = solve_qp(
        P=np.eye(B.shape[0]), 
        q=-eigvals, 
        G=None, h=None,  # No inequality constraints
        A=np.ones(B.shape[0]).reshape(1, -1), 
        b=np.array([1]),  # Sum of weights constraint
        lb=np.zeros(B.shape[0]), 
        ub=np.ones(B.shape[0]), 
        solver="proxqp"
    )
    end = time.time()
    if verbose:
        print(f"{end - start} seconds elapsed for quadratic programming solution")
        
    B_tilde = eigvecs @ np.diag(w) @ eigvecs.T
    
    if lam > 0:
        for r in range(B_tilde.shape[0]):
            for s in range(B_tilde.shape[1]):
                B_tilde[r][s] = np.sign(B_tilde[r][s]) * np.maximum(0, np.abs(B_tilde[r][s] - d * lam))

                
    return B_tilde