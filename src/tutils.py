import numpy as np
from numpy.linalg import qr, svd, norm
import pdb

# Tensor utils using numpy (might be a
# better way to make this efficient)

def tt_oseledets(T, desired_ranks = None, eps = 1e-9):
    '''
    Conversion of a tensor T to tensor train format
    from Oseledets. If rk and eps are set to None then there is no
    tensor train rounding.
    '''
    nks = T.shape
    d = len(nks) # Order of the tensor
    rk_1 = 1
    if desired_ranks is None:
        ranks = np.empty((d-1, ), dtype = int)
    else:
        assert(len(desired_ranks)==d-1, "Desired ranks not the correct size")
        ranks = desired_ranks
    cores = []
    for k in range(d-1):
        # Unfolding
        M = np.reshape(T, (rk_1*nks[k], -1 ))
        # SVD of M
        U, S, Vh = svd(M, full_matrices = False)
        #breakpoint()
        # If possible, truncate
        rk = np.where(S<eps*S[0])[0]
        if len(rk)==0:
            if desired_ranks is None:
                rk = len(S)
            else:
                rk = desired_ranks[k]
        else:
            rk = rk[0]
        if k < d-1:
            ranks[k] = rk
        U, S, Vh = U[:, 0:rk], S[0:rk], Vh[0:rk, :]
        # Set the first core
        y = np.reshape(U, (rk_1,nks[k], rk))
        cores += [y]
        #breakpoint()
        # Define the remainder
        #breakpoint()
        T = np.reshape( S[:, None]*Vh, (rk, *nks[k+1:]) )
        rk_1 = rk
    cores += [np.reshape(T, (T.shape[0], T.shape[1], 1))]
    return cores, ranks


def reconstructFromTT(cores):
    '''
    Given a tensor in TT format reconstruct it
    '''
    d = len(cores)
    nks = [cores[i].shape[1] for i in range(d)]
    if( cores[-1].shape[-1] != 1):
        nks += [cores[-1].shape[-1]]
    T = cores[-1]
    for k in range(d-2, -1, -1):
        # Start from the right of the tensor
        corek = cores[k]
        rk_1, nk, rk = corek.shape
        M = np.reshape(T, (rk, -1))
        M = np.reshape(corek, (rk_1*nk, rk))@M
        #breakpoint()
        T = np.reshape(M, (rk_1, nk, *nks[k+1:]))
    if( cores[0].shape[0] != 1):
        nks = [cores[0].shape[0]] + nks
    T = np.reshape(T, nks)
    return T

def leftOrthTT(cores, ranks, l = None):
    '''
    Given a tensor in TT format and a core index l
    compute the left orthogonalization up to the core l
    '''
    d = len(cores)
    if l is None:
        l = d-1
    nks = [cores[i].shape[1] for i in range(d)]
    ranks = list(ranks) + [1]
    rk_1 = 1
    assert(l<d, "Cant perform left orthogonalization on more than d-1 cores")
    for k in range(l):
        # Compute QR factorization
        Ck = cores[k]
        rk = ranks[k]
        nk = nks[k]
        Q, R = qr(np.reshape(Ck, (rk_1*nk, rk)))
        y = np.reshape(Q, (rk_1, nk, -1))
        s = y.shape[-1]
        z = R@np.reshape( cores[k+1], (rk, nks[k+1]*ranks[k+1]) )
        z = np.reshape(z, (s, nks[k+1], ranks[k+1]))
        cores[k] = y
        cores[k+1] = z
        ranks[k] = s
        rk_1 = s
    ranks.pop()
    return cores, np.array(ranks)
        

def rightOrthTT(cores, ranks, l = None):
    '''
    Given a tensor in TT format and a core index l
    compute the right orthogonalization of the cores l-d
    '''
    d = len(cores)
    if l is None:
        l = d-1
    nks = [cores[i].shape[1] for i in range(d)]
    ranks = list(ranks) + [1]
    rk_1 = 1
    assert(l>0, "Cant perform left orthogonalization on more than d-1 cores")
    for k in range(d-1, l-1, -1):
        Ck = cores[k]
        rk = ranks[k]
        nk = nks[k]
        Rt, Qt = qr(np.reshape(Ck, (ranks[k-1], nk*rk) ) )
        y = np.reshape(Qt, (-1, nk, rk))
        s = y.shape[0]
        z = np.reshape( cores[k-1], (-1, ranks[k-1]))@Rt
        z = np.reshape(z, (-1, nks[k-1], s))
        cores[k] = y
        cores[k-1] = z
        ranks[k-1] = s
    ranks.pop()
    return cores, np.array(ranks)

def pseudoLInverseTT(cores, ranks, l = None, tol = 1e-9):
    '''
    As in Klus, Gelss, Peitz, Schutte algorithm 6
    the pseudo inverse of a matricization of a TT tensor
    '''
    d = len(cores)
    if l is None:
        l = d-2
    nks = [cores[i].shape[1] for i in range(d)]
    ranks = list(ranks) + [1]
    rk_1 = 1
    assert(l<d-1, "Cant perform pseudo inverse for l > d-1")
    # Step 1: left and right orthogonalization
    cores, ranks = leftOrthTT(cores, ranks, l = l-1)
    cores, ranks = rightOrthTT(cores, ranks, l = l+1)
    # Step 2: SVD (truncated)
    U, S, Vh = svd( np.reshape(cores[l], (ranks[l-1]*nks[l], ranks[l])), full_matrices = False)
    s = np.where(S<tol)[0]
    if(len(s)==0):
        s = len(S)
    else:
        s = s[0]
    U, S, Vh = U[:, 0:s], S[0:s], Vh[0:s, :]
    # Reshaped version of U
    y = np.reshape(U, (ranks[l-1], nks[l], -1))
    z = Vh@np.reshape( cores[l+1], (ranks[l], nks[l+1]*ranks[l+1]) )
    z = np.reshape(z, (s, nks[l+1], ranks[l+1]))
    cores[l] = y
    cores[l+1] = z
    ranks[l] = s
    ## Uncomment if we want to contract the cores for the pseudo inverse, ill adviced to do this
    # M = reconstructFromTT([cores[i] for i in range(l+1)])
    # N = reconstructFromTT([cores[i] for i in range(d-1, l, -1)])
    Sinv = [1/si if si>tol else 0 for si in S]
    # Return the pseudo inverse in this almost TT format
    return cores, ranks, Sinv


def dotProdTT(coresA, coresB):
    '''
    Algorithm 4 in Oseledets. Tensor A in TT-format with
    cores coresA, tensor B in TT-format with cores coresB
    outputs <A,B>
    '''
    d = len(coresA)
    assert(d == len(coresB), "Tensors of incompatible size to compute dot product")
    Ak = coresA[0]
    Bk = coresB[0]
    nksA = [coresA[i].shape[1] for i in range(d)]
    nksB = [coresA[i].shape[1] for i in range(d)]
    assert( np.all(nksA == nksB), "Incompatible tensor dimensions")
    Ak = coresA[0]
    Bk = coresB[0]
    v = np.kron( Ak[:, 0, :], Bk[:, 0, :])
    for i in range(1, nksA[0]):
        v += np.kron( Ak[:, i, :], Bk[:, i, :])
    v = np.sum( [np.kron(Ak[:, i, :], Bk[:, i, :]) for i in range(nksA[0])], axis = 0)
    for k in range(1,d):
        Ak = coresA[k]
        Bk = coresB[k]
        # Compute gammak
        gammak = np.kron( Ak[:, 0, :], Bk[:, 0, :])
        for i in range(1, nksA[k]):
            gammak += np.kron( Ak[:, i, :], Bk[:, i, :])
        v = v@gammak
    v = np.reshape(v, (coresA[-1].shape[-1], coresB[-1].shape[-1]))
    return v


def transposeTT(cores, ranks):
    '''
    Given a tensor in TT format transpose it
    '''
    ranks = np.flip(ranks)
    cores = cores[::-1]
    for k in range(len(cores)):
        cores[k] = cores[k].T
    return cores, ranks


def tt_ice(cores, ranks, Y, tol = 1e-9):
    '''
    Algorithm 3.1, TT-ICE from
    Aksoy, Gorsich, Veerapaneni, Gorodetsky
    "An incremental tensor train decomposition algorithm"
    Incrementally updates the tt decomposition of a stream of
    tensor data.
    '''
    # Preprocessing the first dimension
    d = len(cores)
    cores_new = []
    ranks_new = ranks.copy()
    assert( d == len(Y), "Incompatible order of tensors")
    nks = [cores[i].shape[1] for i in range(d)]
    Yi = np.reshape(Y, (nks[0], -1))
    Ui = np.reshape( cores[0], (nks[0], ranks[0]) )
    Ri = (np.eye(Ui.shape[0]) - Ui@Ui.T)@Yi
    Ui_pad = Ui # First core has no padding
    # Update the cores
    for i in range(d-2):
        if( norm(Ri) > tol):
            U, S, Vh = svd(Ri, full_matrices = False)
            # Truncate
            s = np.where(S<tol)[0]
            if(len(s)==0):
                s = len(S)
            else:
                s = s[0]
            U, S, Vh = U[:, 0:s], S[0:s], Vh[0:s, :]
            Ui = np.concatenate((Ui_pad, U), axis = 1 )
        else:
            Ui = Ui_pad
            s = 0
        # Save
        ranks_new[i] += s
        cores_new += [ np.reshape(Ui, (-1, nks[i], ranks_new[i] ) ) ]
        # Pad for rank compatibility
        Ui = np.reshape( cores[i+1], (ranks[i]*nks[i+1], -1) )
        Ui_pad = np.reshape( np.concatenate((np.reshape(Ui, (ranks[i], -1)), np.zeros((s, nks[i+1]*ranks[i+1])) ), axis = 0), (ranks_new[i]*nks[i+1], ranks[i+1]) )
        # Preprocessing the subsequent dimensions
        Yi = np.reshape( np.reshape(cores_new[i], (-1, ranks_new[i])).T@Yi, (ranks_new[i]*nks[i+1], -1))
        Ri = Yi - Ui_pad@(Ui_pad.T@Yi)
        #Ri = (np.eye(Ui_pad.shape[0]) - Ui_pad@Ui_pad.T)@Yi
    # Updating the d-th core
    if( norm(Ri)>tol):
        U, S, Vh = svd(Ri, full_matrices = False)
        # Truncate
        s = np.where(S<tol)[0]
        if(len(s) == 0):
            s = len(S)
        else:
            s = s[0]
        U, S, Vh = U[:, 0:s], S[0:s], Vh[0:s, :]
        Ui = np.concatenate((Ui_pad, U), axis = 1)
    else:
        Ui = Ui_pad
        s = 0
    ranks_new[-1] += s
    cores_new += [ np.reshape( Ui, (ranks_new[-2], nks[-2], ranks_new[-1] )  )]
    # Update the last core
    Ui_pad = np.concatenate((np.reshape(cores[-1], (ranks[-1], -1)), np.zeros((s, nks[-1])) ), axis = 0)
    Yi = Ui.T@Yi
    Ui = np.concatenate( (Ui_pad, Yi), axis = 1)
    Ui = np.reshape(Ui, (ranks_new[-1], -1, 1))
    cores_new += [ Ui ]
    return cores_new, ranks_new
    
    

    
