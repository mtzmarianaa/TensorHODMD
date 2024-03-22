# Simple DMD code

import numpy as np
from numpy.linalg import svd, qr, eig
import matplotlib.pyplot as plt
import colorcet as cc
import scipy.sparse.linalg
import pdb


def DMD_Schmid(Xm, Ym, k = None):
    '''
    Schmid's DMD as in slide 68 from the
    course Numerical linear algebra for
    Koopman and DMD.
    In:
       - Xm (n,m) matrix that defines a sequence of snapshots
       - Ym (n,m) matrix that defines a sequence of snapshots
       - k  truncation parameter, numerical rank
    Assumption: m<<n
    Out:
       - Zk  Ritz vectors
       - Lamk Eigenvalues
    '''
    m, n = Xm.shape
    assert(Xm.shape == Ym.shape, "Snapshot matrices should be the same size")
    assert(m < n, "m should be smaller than n")
    # Step 1: thin SVD
    # TEST WHICH IS FASTER, WITH OR WITHOUT QR
    Q, R = qr(Xm)
    U, S, Vh = svd(R, full_matrices = False) # Reduced SVD
    # Determine the numerical rank heuristically for now
    if(k is None or k>len(S)):
        k = 10
        k = np.min([k, len(S), m/2])
    # Truncate
    U, S, Vh = U[:, 0:k], S[0:k], Vh[0:k, :]
    # Schmid's formula for the Rayleigh quotient
    U = Q@U
    Sk = ( (U.conj().T@Ym)@Vh.conj().T  )
    Sk = np.dot(Sk, 1/S)
    # Eigenvalues of the Rayleigh quotient
    Lamk, Wk = eig(Sk)
    # Get Ritz vectors
    Zk = U@Wk
    return Zk, Lamk
    
    

