import numpy as np
from numpy.linalg import qr, svd, norm
from tutils import *


def getACom(X, Y, desired_ranks = None, eps = 1e-9):
    '''
    Given snapshots in tensor form we compute
    A_tilde = MT@P@Q@NT@Sigma-1 from p. 11
    of Klus, Gelss, Peitz, Schutte
    '''
    # First get X, Y in tensor representation
    nksX = X.shape
    nksY = Y.shape
    dd = len(nksX)
    assert( dd == len(nksY), "Snapshots tensors of different order")
    assert( np.all(nksX) == np.all(nksY), "Snapshot tensors of different size")
    coresX, ranksX = tt_oseledets(X, desired_ranks = desired_ranks, eps = eps)
    coresY, ranksY = tt_oseledets(Y, desired_ranks = desired_ranks, eps = eps)
    # Get pseudo inverse
    coresXpi, rankspi, Sinv = pseudoLInverseTT(coresX, ranksX, tol = eps)
    # Compute dot product of two almost tt tensors
    MTP = dotProdTT( coresXpi[0:dd-1], coresY[0:dd-1] )
    Q = coresY[-1]
    if(len(Q.shape)>2):
        Q = np.reshape(Q, (Q.shape[0], Q.shape[1]))
    N = coresXpi[-1]
    if(len(N.shape)>2):
        N = np.reshape(N, (N.shape[0], N.shape[1]))
    NT = N.T
    Sinv = np.array(Sinv)
    Atilde = (MTP@Q@NT)*Sinv[:, np.newaxis]
    print("Shape of compressed operator: ", Atilde.shape)
    return Atilde

