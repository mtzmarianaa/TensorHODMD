# Simple DMD code

import numpy as np
from numpy.linalg import svd, qr, eig, norm
import matplotlib.pyplot as plt
import colorcet as cc
import scipy.sparse.linalg
import pdb


def DMD_Schmid(Xm, Ym, k = None, tol = None):
    '''
    Schmid's DMD as in slide 68 from the
    course Numerical linear algebra for
    Koopman and DMD.
    In:
       - Xm (n,m) matrix that defines a sequence of snapshots
       - Ym (n,m) matrix that defines a sequence of snapshots
       - k  truncation parameter, numerical rank
       - tol tolerance for the residual
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
    if((k is None or k>len(S)) and tol is None):
        k = 10
        k = np.min([k, len(S), m/2])
    elif(tol is not None):
        sigma1 = S[0]
        ind = np.where(S<tol*sigma1)[0]
        if(len(ind) == 0):
            k = len(S)
        else:
            k = ind[0]
    else:
        tol = 1e-8
    # Truncate
    k = int(k)
    U, S, Vh = U[:, 0:k], S[0:k], Vh[0:k, :]
    # Schmid's formula for the Rayleigh quotient
    U = Q@U
    Sk = ( (U.conj().T@Ym)@Vh.conj().T  )
    Sinv = np.array([1/s if s>tol else 0 for s in S])
    Sk = Sk*Sinv[:, np.newaxis]
    # Eigenvalues of the Rayleigh quotient
    Lamk, Wk = eig(Sk)
    # Get Ritz vectors
    Zk = U@Wk
    return Zk, Lamk
    
def DMD_Enhanced(Xm, Ym, k = None, tol = 1e-8):
    '''
    Enhanced DMD as in lecture notes from the couse
    Numerical linear algebra for Koopman and DMD.
    In:
       - Xm (n,m) matrix that defines a sequence of snapshots
       - Ym (n,m) matrix that defines a sequence of snapshots
       - k  truncation parameter, numerical rank
       - tol tolerance for the residual
    Assumption: m<<n
    Out:
       - Zk  Ritz vectors
       - Lamk Eigenvalues
       - res optimal residuals found
    '''
    m, n = Xm.shape
    assert(Xm.shape == Ym.shape, "Snapshot matrices should be the same size")
    assert(m < n, "m should be smaller than n")
    assert(tol>0, "tolerance should be positive")
    # Step 1: divide by Xm's norm of its columns
    Dx = [norm(Xm[:, i]) for i in range(n)]
    Dx = [1/d if d>tol else 0 for d in Dx]
    Xm = Xm*Dx
    Ym = Ym*Dx
    # Step 2: SVD of Xm
    U, S, Vh = svd(Xm, full_matrices = False) # Reduced SVD
    # Determine the numerical rank heuristically for now
    if(k is None):
        sigma1 = S[0]
        ind = np.where(S<tol*sigma1)[0]
        if(len(ind) == 0):
            k = len(S)
        else:
            k = ind[0]
    # Truncate
    U, S, Vh = U[:, 0:k], S[0:k], Vh[0:k, :]
    # Step 3: Schmid's formula for the Rayleigh quotient
    Bk = Ym@(Vh.conj().T*1/S)
    # Step 4: Thin QR factorization
    Q, R = qr( np.hstack((U, Bk)) )
    # Step 5: Sk = Uk.T A Uk
    rii = np.diagonal(R)[0:k].conj().reshape(-1,1)
    Sk = R[0:k, k:2*k]*rii
    Lamk, Wk = eig(Sk)
    res = np.empty((k,))
    # Step 6: Minimize the residuals
    for i in range(k):
        Rlam = np.vstack((R[0:k, k:2*k] - Lamk[i]*R[0:k, 0:k], R[k:2*k, k:2*k]))
        _, SigmaLam, WLam = svd( Rlam )
        WLam = WLam.T
        # Get the smallest sigma, w
        indSmallest = max(0, np.where(SigmaLam<tol)[0])
        Wk[:, i] = WLam[indSmallest-1]
        res[i] = SigmaLam[indSmallest]
        #print("Residual: ", res[i])
    Zk = U@Wk
    return Zk, Lamk, res


def DMD_QR(Xm, Ym, k = None, tol = 1e-8):
    '''
    QR compressed DMD as in lecture notes from the couse
    Numerical linear algebra for Koopman and DMD.
    In:
       - Xm (n,m) matrix that defines a sequence of snapshots
       - Ym (n,m) matrix that defines a sequence of snapshots
       - k  truncation parameter, numerical rank
       - tol tolerance for the residual
    Assumption: m<<n
    Out:
       - Zk  Ritz vectors
       - Lamk Eigenvalues
    '''
    m, n = Xm.shape
    assert(Xm.shape == Ym.shape, "Snapshot matrices should be the same size")
    assert(m < n, "m should be smaller than n")
    assert(tol>0, "tolerance should be positive")
    X = np.empty((m, n+1))
    X[:, 0:n] = Xm
    X[:, -1] = Ym[:, -1]
    Q, R = qr(X) # Thin QR factorization, compressed representation of the data
    Rx = R[:, 0:n]
    Ry = R[:, 1:]
    Zk, Lamk = DMD_Schmid(Rx, Ry, k = k, tol = tol)
    Zk = Q@Zk
    return Zk, Lamk
    

    
