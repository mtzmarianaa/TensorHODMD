import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.tri as tri
from src.tutils import *
from src.tensorDMD import *
from src.DMDSimply import *
import pickle
import timeit

plt.ion()


# Load the triangulation info
colormap  = "cet_linear_worb_100_25_c53"
path_info = "tests/TestEikonal/TestEikonal/"
H = "H1"
eik_coords = np.genfromtxt(path_info + H + "/" + H + "_MeshPoints.txt", delimiter=",")
triangles_points = np.genfromtxt(path_info + H + "/" + H + "_Faces.txt", delimiter=",")
triang = tri.Triangulation(eik_coords[:, 0], eik_coords[:, 1], triangles_points)
nPoints = len(eik_coords)
nPlot = 4
my_dpi=96


# Read the data
file = open('TX', 'rb')
TX = pickle.load(file)
file.close()

# For different amounts of snapshots
nSnapshots = np.arange(20,40)
timeTDMD = np.empty((20, ))
errTDMD = np.empty((20, ))
timeDMD = np.empty((20, ))
errDMD = np.empty((20, ))
timeEDMD = np.empty((20, ))
errEDMD = np.empty((20, ))
timeQRDMD = np.empty((20, ))
errQRDMD = np.empty((20, ))
sizeX = np.zeros((20, ))
sizeCompX = np.zeros((20, ))

for mi in range(20):
    m = nSnapshots[mi]
    # Divide into X and Y
    ds = TX.shape
    d1 = len(ds)
    X = TX[:, :, :, 0:(m-1)]
    Y = TX[:, :, :, 1:m]

    # Compute the TDMD nodes
    coresX, ranksX = tt_oseledets(X)
    coresY, ranksY = tt_oseledets(Y)
    start = timeit.default_timer()
    Lamk, phi, coresX, coresY =  tensorDMD(coresX = coresX, ranksX = ranksX,
                                           coresY = coresY, ranksY = ranksY)
    stop = timeit.default_timer()
    timeTDMD[mi] = stop - start
    LamkE, phiE, coresXE, coresYE = tensorDMD(X, Y, eps = 1e-16)
    errTDMD[mi] = norm(Lamk - LamkE)/norm(LamkE)

    # Get sizes
    sizeX[mi] = X.size
    for sz in range(4):
        sizeCompX[mi] += coresX[sz].size

    # Now reshape data, use matrix DMD
    Xm = np.reshape(X, (ds[0]*ds[1]*ds[2], -1))
    Ym = np.reshape(Y, (ds[0]*ds[1]*ds[2], -1))
    start = timeit.default_timer()
    Zk, Lamk = DMD_Schmid(Xm, Ym, tol = 1e-9)
    stop = timeit.default_timer()
    timeDMD[mi] = stop - start
    Zk = np.real(Zk)
    ZkE, LamkE = DMD_Schmid(Xm, Ym, tol = 1e-16)
    errDMD[mi] = norm(Lamk - LamkE)/norm(LamkE)
    # Enhanced DMD
    start = timeit.default_timer()
    ZkRef, LamkRef, res = DMD_Enhanced(Xm, Ym, tol = 1e-9)
    stop = timeit.default_timer()
    timeEDMD[mi] = stop - start
    ZkRef = np.real(ZkRef)
    ZkRefE, LamkRefE, resE = DMD_Enhanced(Xm, Ym, tol = 1e-16)
    errEDMD[mi] = norm(LamkRef - LamkRefE)/norm(LamkRefE)
    # Compressed DMD
    start = timeit.default_timer()
    ZkCom, LamkCom = DMD_QR(Xm, Ym, tol = 1e-9)
    stop = timeit.default_timer()
    timeQRDMD[mi] = stop - start
    ZkCom = np.real(ZkCom)
    ZkComE, LamkComE = DMD_QR(Xm, Ym, tol = 1e-16)
    errQRDMD[mi] = norm(LamkCom - LamkComE)/norm(LamkComE)

    # Reshape accordingly
    Zk = np.reshape(Zk, (X.shape))
    ZkRef = np.reshape(ZkRef, (X.shape))
    ZkCom = np.reshape(ZkCom, (X.shape))

    #Plot
    for i in range(nPlot):
        ### TENSOR DMD
        # Get the TDMD mode
        lambd = Lamk[i]
        wi = phi[-1][:, i, :]
        wi = np.reshape(wi, (*wi.shape, 1))
        # Get phi_i as a full tensor
        phi_i = phi[0:3] + [wi]
        dmdMode = reconstructFromTT(phi_i)
        # Plot the information corresponding to the eikonal
        eikdmdMode = dmdMode[0, :, 0, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = eikdmdMode[:,0], cmap = colormap)
        plt.colorbar(im)
        plt.title("TDMD mode for eikonal value with $\| \lambda \|=$" + '{}'.format(round(norm(lambd), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the gradients
        graddmdMode = dmdMode[0, :, 1:3, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = eikdmdMode[:,0], cmap = colormap)
        plt.quiver( eik_coords[:, 0], eik_coords[:, 1], graddmdMode[:, 0], graddmdMode[:,1], scale = 30  )
        plt.colorbar(im)
        plt.title("TDMD mode for eikonal value and gradients with $\| \lambda \|=$" + '{}'.format(round(norm(lambd), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the type of solution
        typeSoldmdMode = dmdMode[0, :, -1, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = typeSoldmdMode[:,0], cmap = colormap)
        plt.colorbar(im)
        plt.title("TDMD mode for type of solution with $\| \lambda \|=$" + '{}'.format(round(norm(lambd), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        ### DMD
        # Get the DMD mode
        lamDMD = Lamk[i]
        zk = Zk[0, :, 0, i]
        lamEDMD = LamkRef[i]
        zkEDMD = ZkRef[0, :, 0, i]
        lamQRDMD = LamkCom[i]
        zkQRDMD = ZkCom[0, :, 0, i]
        eikdmdMode = dmdMode[0, :, 0, :]
        # Plot the information corresponding to the eikonal
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = zk, cmap = colormap)
        plt.colorbar(im)
        plt.title("DMD mode for eikonal value with $\| \lambda \|=$" + '{}'.format(round(norm(lamDMD), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the type of solution
        typeSoldmdMode = dmdMode[0, :, -1, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = Zk[0, :, -1, i], cmap = colormap)
        plt.colorbar(im)
        plt.title("DMD mode for type of solution with $\lambda =$" + '{}'.format(round(norm(lamDMD), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the eikonal
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = zkEDMD, cmap = colormap)
        plt.colorbar(im)
        plt.title("Refined DMD mode for eikonal value with $\| \lambda \|=$" + '{}'.format(round(norm(lamEDMD), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the type of solution
        typeSoldmdMode = dmdMode[0, :, -1, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkRef[0, :, -1, i], cmap = colormap)
        plt.colorbar(im)
        plt.title("Refined DMD mode for type of solution with $\| \lambda \|=$" + '{}'.format(round(norm(lamEDMD), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the eikonal
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = zkQRDMD, cmap = colormap)
        plt.colorbar(im)
        plt.title("QRDMD mode for eikonal value with $\| \lambda \|=$" + '{}'.format(round(norm(lamQRDMD), 3)))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        # Plot the information corresponding to the type of solution
        typeSoldmdMode = dmdMode[0, :, -1, :]
        fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        im = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkCom[0, :, -1, i], cmap = colormap)
        plt.colorbar(im)
        plt.title("QRDMD mode for type of solution with $\|\lambda \|=$" + '{}'.format(round(norm(lamQRDMD), 3) ))
        ax = plt.gca()
        ax.set_aspect('equal')
        ax.set_xlim([-18,18])
        ax.set_ylim([-18,24])
        
# Plot times, errors        
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(sizeX, timeTDMD, c = "#b200ff", label = "TDMD")
plt.loglog(sizeX, timeDMD, c = "#0004ff", label = "DMD")
plt.loglog(sizeX, timeEDMD, c = "#00aeff", label = "EDMD")
plt.loglog(sizeX, timeQRDMD, c = "#00fff0", label = "QRDMD")
plt.title("Size of X, time taken to get DMD modes")
plt.xlabel("Size of X")
plt.ylabel("Time")
plt.legend()

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(sizeX, errTDMD, c = "#b200ff", label = "TDMD")
plt.loglog(sizeX, errDMD, c = "#0004ff", label = "DMD")
plt.loglog(sizeX, errEDMD, c = "#00aeff", label = "EDMD")
plt.loglog(sizeX, errQRDMD, c = "#00fff0", label = "QRDMD")
plt.title("Size of X, errors in DMD modes")
plt.xlabel("Size of X")
plt.ylabel("Error in DMD modes")
plt.legend()

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(sizeX, sizeCompX, c = "#410093")
plt.title("Size of $\mathcal{X}$ and size of its TT representation")
plt.xlabel("Size of X")
plt.ylabel("Size of TT representation of X")

    
plt.show()


