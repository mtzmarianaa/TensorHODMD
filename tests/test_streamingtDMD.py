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
timeStDMD = np.empty((20,))
errStDMD = np.empty((20, ))
sizeX = np.zeros((20, ))
sizeCompX = np.zeros((20, ))

XBase = TX[:, :, :, 0:19]
YBase = TX[:, :, :, 1:20]
coresXBase, ranksXBase = tt_oseledets(XBase)
coresYBase, ranksYBase = tt_oseledets(YBase)

for mi in range(20):
    m = nSnapshots[mi]
    # Divide into X and Y
    ds = TX.shape
    d1 = len(ds)
    X = TX[:, :, :, 0:(m-1)]
    Y = TX[:, :, :, 1:m]
    XNew = TX[:, :, :, 19:(m-1)]
    YNew = TX[:, :, :, 20:m]

    # Compute the TDMD nodes
    start = timeit.default_timer()
    coresX, ranksX = tt_oseledets(X)
    coresY, ranksY = tt_oseledets(Y)
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

    # Compute the streamign tDMD mdoes
    start = timeit.default_timer()
    LamkS, phiS, coresXS, coresYS = streamingTensorDMD(coresX = coresXBase, ranksX = ranksXBase,
                                                       coresY = coresYBase, ranksY = ranksYBase,
                                                       XNew = XNew, YNew = YNew)
    stop = timeit.default_timer()
    timeStDMD[mi] = stop - start
    errStDMD[mi] = norm(LamkS - LamkE)/norm(LamkE)


        
# Plot times, errors        
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(sizeX, timeTDMD, c = "#b200ff", label = "TDMD")
plt.loglog(sizeX, timeStDMD, c = "#0004ff", label = "sTDMD")
plt.title("Size of X, time taken to get DMD modes")
plt.xlabel("Size of X")
plt.ylabel("Time")
plt.legend()

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(sizeX, errTDMD, c = "#b200ff", label = "TDMD")
plt.loglog(sizeX, errStDMD, c = "#0004ff", label = "sDMD")
plt.title("Size of X, errors in DMD modes")
plt.xlabel("Size of X")
plt.ylabel("Error in DMD modes")
plt.legend()


lenStream = np.arange(1, 21)
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.loglog(lenStream, timeTDMD, c = "#b200ff", label = "TDMD")
plt.loglog(lenStream, timeStDMD, c = "#0004ff", label = "sTDMD")
plt.title("Length of stream, time taken to get DMD modes")
plt.xlabel("Length of stream")
plt.ylabel("Time")
plt.legend()

    
plt.show()
