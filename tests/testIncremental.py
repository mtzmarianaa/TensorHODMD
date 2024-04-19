import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.tri as tri
from src.tutils import *
from src.tensorDMD import *
import pickle

plt.ion()

# Load the triangulation info
colormap  = "cet_linear_worb_100_25_c53"
path_info = "tests/TestEikonal/TestEikonal/"
H = "H1"
eik_coords = np.genfromtxt(path_info + H + "/" + H + "_MeshPoints.txt", delimiter=",")
triangles_points = np.genfromtxt(path_info + H + "/" + H + "_Faces.txt", delimiter=",")
triang = tri.Triangulation(eik_coords[:, 0], eik_coords[:, 1], triangles_points)
nPoints = len(eik_coords)
my_dpi=96


# Read the data
file = open('TX', 'rb')
TX = pickle.load(file)
file.close()

X = TX[:, :, :, 0:30]

Xadd = TX[:, :, :, 30:32]
Xnew = TX[:, :, :, 0:32]

cores, ranks = tt_oseledets(X, eps = 1e-5)
coresNew, ranksNew = tt_ice(cores, ranks, Xadd, tol = 1e-5)
coresDirect, ranksDirect = tt_oseledets(Xnew, eps = 1e-5)

print("\nRunning all tests with tol = 1e-5\n")

print("\nDimensions of cores, X")
for i in range(4):
    print(cores[i].shape)

print("\nDimensions of new accumulation")
for i in range(4):
    print(coresNew[i].shape)

print("\nIf done tt oseledets again on accumulation tensor, dimensions:")
for i in range(4):
    print(coresDirect[i].shape)

print("\nAbsolute error of tt_ice:")
XiceNew = reconstructFromTT(coresNew)
err_ice = norm( XiceNew - Xnew)
print(err_ice)

print("\nRelative error of tt_ice:")
print(err_ice/norm(Xnew))
