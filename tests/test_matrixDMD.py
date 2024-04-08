import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.tri as tri
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


