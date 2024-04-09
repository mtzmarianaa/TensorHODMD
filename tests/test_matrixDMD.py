import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.tri as tri
from src.DMDSimply import *
plt.ion()



colormap2  = "cet_linear_worb_100_25_c53_r"


#colormap3  = "cet_diverging_cwm_80_100_c22"
colormap3 = cm.get_cmap('PiYG', 5)


colormap4  = "cet_linear_worb_100_25_c53"

colormap5  = "cet_linear_worb_100_25_c53"


my_dpi=96
eta1 = 1.0
eta2s = np.linspace(0.8, 1.5, 50)
#eta2s = np.append(eta2s, eta2s - 0.8 + 1.5)
x0_default = np.array([-15, -10])
center_default = np.array([0,0])
R_default = 10.0
path_data = "tests/TestEikonal/"
path_info = "tests/TestEikonal/TestEikonal/"
H = "H1"


# Import the triangulation info
eik_coords = np.genfromtxt(path_info + H + "/" + H + "_MeshPoints.txt", delimiter=",")
triangles_points = np.genfromtxt(path_info + H + "/" + H + "_Faces.txt", delimiter=",")
triang = tri.Triangulation(eik_coords[:, 0], eik_coords[:, 1], triangles_points)
nPoints = len(eik_coords)

# Load data - first only the eikonal at each point
eikData = np.genfromtxt(path_data + H + "_snapshots_true_values.txt", delimiter = ",")
# Divide data into Xm and Ym

Xm = eikData[:, 0:-1]
Ym = eikData[:, 1:]

# Normal DMD
Zk, Lamk = DMD_Schmid(Xm, Ym, tol = 1e-9)
Zk = np.real(Zk)
# Enhanced DMD
ZkRef, LamkRef, res = DMD_Enhanced(Xm, Ym, tol = 1e-9)
ZkRef = np.real(ZkRef)
# Compressed DMD
ZkCom, LamkCom = DMD_QR(Xm, Ym, tol = 1e-9)
ZkCom = np.real(ZkCom)

#####
# Plot the Zk
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im1 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = Zk[:,0], cmap = colormap2)
plt.colorbar(im1)
plt.title("First Zk for DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im2 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = Zk[:,1], cmap = colormap2)
plt.colorbar(im2)
plt.title("Second Zk for DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im3 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = Zk[:,2], cmap = colormap2)
plt.colorbar(im3)
plt.title("Third Zk for DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])


#####
# Plot the ZkRef
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im4 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkRef[:,0], cmap = colormap2)
plt.colorbar(im4)
plt.title("First ZkRef for Refined DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im5 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkRef[:,1], cmap = colormap2)
plt.colorbar(im5)
plt.title("Second ZkRef for DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im6 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkRef[:,2], cmap = colormap2)
plt.colorbar(im6)
plt.title("Third ZkRef for Refined DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])


#####
# Plot the ZkCom
fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im7 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkCom[:,0], cmap = colormap2)
plt.colorbar(im7)
plt.title("First ZkCom for DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im8 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkCom[:,1], cmap = colormap2)
plt.colorbar(im8)
plt.title("Second ZkCom for QR Compressed DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
im9 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = ZkCom[:,2], cmap = colormap2)
plt.colorbar(im9)
plt.title("Third ZkCom for QR Compressed DMD")
ax = plt.gca()
ax.set_aspect('equal')
ax.set_xlim([-18,18])
ax.set_ylim([-18,24])

