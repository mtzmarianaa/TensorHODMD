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

# Divide into X and Y
ds = TX.shape
d1 = len(ds)
m = ds[-1]
X = TX[:, :, :, 0:(m-1)]
Y = TX[:, :, :, 1:]

# Compute the TDMD nodes
Lamk, phi, coresX, coresY =  tensorDMD(X, Y)

# Plot

nPlot = 4

for i in range(nPlot):
    # Get the DMD mode
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
    plt.title("TDMD mode for eikonal value with $\lambda =$" + '{}'.format(lambd))
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
    plt.title("TDMD mode for eikonal value and gradients with $\lambda =$" + '{}'.format(lambd))
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
    plt.title("TDMD mode for type of solution with $\lambda =$" + '{}'.format(lambd))
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_xlim([-18,18])
    ax.set_ylim([-18,24])

plt.show()
