import numpy as np
from numpy.linalg import qr, svd, norm, eig
from tutils import *
import pdb

# For tensor DMD we first need the reduced matrix A tilde
def getACom(X = None, Y = None, coresX = None,
            ranksX = None, coresY = None, ranksY = None,
            desired_ranks = None, eps = 1e-9):
    '''
    Given snapshots in tensor form we compute
    A_tilde = MT@P@Q@NT@Sigma-1 from p. 11
    of Klus, Gelss, Peitz, Schutte
    '''
    if(coresX is None):
        # First get X, Y in tensor representation if not given
        nksX = X.shape
        nksY = Y.shape
        dd = len(nksX)
        assert( dd == len(nksY), "Snapshots tensors of different order")
        assert( np.all(nksX) == np.all(nksY), "Snapshot tensors of different size")
        coresX, ranksX = tt_oseledets(X, desired_ranks = desired_ranks, eps = eps)
        coresY, ranksY = tt_oseledets(Y, desired_ranks = desired_ranks, eps = eps)
    else:
        dd = len(coresX)
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
    return Atilde, coresXpi, rankspi, Sinv

# Now we get the DMD modes
def tensorDMD(X = None, Y = None, coresX = None,
              ranksX = None, coresY = None, ranksY = None,
              desired_ranks = None, eps = 1e-9, Atilde = None,
              coresXpi = None, rankspi = None, Sinv = None):
    assert(X is not None or Atilde is not None, "Give either X or Atilde")
    # TT format if not given
    if(coresX is None):
        nksX = X.shape
        nksY = Y.shape
        dd = len(nksX)
        assert( dd == len(nksY), "Snapshots tensors of different order")
        assert( np.all(nksX) == np.all(nksY), "Snapshot tensors of different size")
        coresX, ranksX = tt_oseledets(X, desired_ranks = desired_ranks, eps = eps)
        coresY, ranksY = tt_oseledets(Y, desired_ranks = desired_ranks, eps = eps)
    dd = len(coresX)
    # Compute Atilde if not given
    if(Atilde is None):
        Atilde, coresXpi, rankspi, Sinv = getACom(X, Y, coresX,
                                                  ranksX, coresY, ranksY,
                                                  desired_ranks, eps)
    # Compute eigenvalues
    Lamk, Wk = eig(Atilde)
    # Then get the DMD modes in a TT-representation
    MT = coresXpi[0:dd-1]
    # Contract the last core of MT with the matrix of Eigenvalues
    # but we don't really contract it, we just represent the
    # DMD modes in TT form
    Wk = np.reshape(Wk, (*Wk.shape, 1))
    phi = MT + [Wk]
    # Return the eigenvalues
    return Lamk, phi, coresX, coresY

