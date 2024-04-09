## Plot snapshots from the Eikonal equation
# Given a specific triangulation, vary the ratio
# in the indices of refraction. Each different ratio corresponds
# to a different snapshot. Notice that we are using unstructured
# meshes for this.

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import colorcet as cc
from analyticSol_circle import trueSolution
import pdb
from matplotlib import cm
import matplotlib.animation as animation
plt.ion()


# Previous useful definitions

colormap2  = "cet_linear_worb_100_25_c53_r"


#colormap3  = "cet_diverging_cwm_80_100_c22"
colormap3 = cm.get_cmap('PiYG', 5)


colormap4  = "cet_linear_worb_100_25_c53"

colormap5  = "cet_linear_worb_100_25_c53"


spacingGrid = 25
nx_default = 36*spacingGrid
ny_default = 42*spacingGrid
my_dpi=96
eta1 = 1.0
eta2s = np.linspace(0.75, 1.5, 40)
#eta2s = np.append(eta2s, eta2s - 0.8 + 1.5)
x0_default = np.array([-15+0.5, -10+0.5])
center_default = np.array([0,0])
R_default = 10.0
path_figures = "tests/TestEikonal/TestEikonal/"
path_info = "tests/TestEikonal/TestEikonal/"
H = "H0"


def plotSnapshots(H, eta1 = eta1, eta2s = eta2s, eik_coords = None, triangles_points = None, x0 = x0_default, center = center_default,
                  R = R_default, saveFigures = True):
    '''
    Given an array of different indices of refraction to try out,
    compute the eikonal equation at a given triangulation
    (square with circle inside). Save those computed eikonal
    values and plots (snapshots)
    '''
    xi, yi = np.meshgrid(np.linspace(-18, 18, nx_default), np.linspace(-18, 24, ny_default))
    xiG, yiG = np.meshgrid(np.linspace(-18, 18, 10), np.linspace(-18, 24, 10))
    # Read the mesh
    if eik_coords is None or triangles_points is None:
        eik_coords = np.genfromtxt(path_info + H + "/" + H + "_MeshPoints.txt", delimiter=",")
        triangles_points = np.genfromtxt(path_info + H + "/" + H + "_Faces.txt", delimiter=",")
    # Define the triangulation
    triang = tri.Triangulation(eik_coords[:, 0], eik_coords[:, 1], triangles_points)
    nPoints = len(eik_coords)
    # For the different eta2 compute the solution and plot
    for k in range(len(eta2s)):
        eta2 = eta2s[k]
        true_sol = np.zeros((nPoints))
        typeSol = np.zeros((nPoints))
        true_grads = np.zeros((nPoints, 2))
        for i in range(nPoints):
            sol, typ, trueGrad = trueSolution( eik_coords[i, 0], eik_coords[i, 1], x0, center, R, eta1, eta2)
            true_sol[i] = sol
            typeSol[i] = typ
            true_grads[i, :] = trueGrad
        # Save
        np.savetxt(path_info + H + "/" + H + "_true_values_" + str(k) + ".txt", true_sol, delimiter = ", ", fmt = "%5.12f")
        np.savetxt(path_info + H + "/" + H + "_typeSol_" + str(k) + ".txt", typeSol, delimiter = ", ")
        np.savetxt(path_info + H + "/" + H + "_true_grads_ " + str(k) + ".txt", true_grads, delimiter = ", ", fmt = "%5.12f")
        # PLOT with linear interpolation
        # interp_lin = tri.LinearTriInterpolator(triang, true_sol)
        # zi_lin = interp_lin(xi, yi)
        # # For the gradients
        # trueGMesh = np.zeros((100, 2))
        # for i in range(10):
        #     for j in range(10):
        #         _, _, gMesh = trueSolution( xiG[i,j], yiG[i,j], x0, center, R, eta1, eta2)
        #         trueGMesh[j + 10*i, 0] = gMesh[0]
        #         trueGMesh[j + 10*i, 1] = gMesh[1]
        # # Points on triangulation and exact eikonal
        # fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        # plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        # im1 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7500/len(eik_coords)), c = true_sol, cmap = colormap2)
        # plt.colorbar(im1)
        # plt.title("Eikonal on triangulation points, ratio=" + '{:.4f}'.format(eta2/eta1))
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_xlim([-18,18])
        # ax.set_ylim([-18,24])
        # if (saveFigures):
        #     plt.savefig(path_figures + H + "/" + H + '_EikonalTriangulation_' + str(k) + '.png', dpi=my_dpi * 10)
        # # Points on triangulation, exact eikonal and gradients
        # fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        # plt.triplot( eik_coords[:, 0], eik_coords[:, 1], triangles_points, '-', c = "#d4bdff", lw = 0.3 )
        # im2 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7000/len(eik_coords)), c = true_sol, cmap = colormap2)
        # plt.quiver(xiG, yiG, trueGMesh[:, 0], trueGMesh[:, 1])
        # plt.colorbar(im2)
        # plt.title("Eikonal on triangulation points and gradients, ratio=" + '{:.4f}'.format(eta2/eta1))
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_xlim([-18,18])
        # ax.set_ylim([-18,24])
        # if (saveFigures):
        #     plt.savefig(path_figures + H + "/" + H + '_EikonalTriangulationGrads_' + str(k) + '.png', dpi=my_dpi * 10)
        # # Level sets of exact solution on triangulation
        # fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        # im3 = plt.contourf(xi, yi, zi_lin, cmap = colormap2, levels = 30 )
        # plt.contour(xi, yi, zi_lin, colors = ["white"], levels = 30)
        # plt.title("Level sets exact solution, ratio=" + '{:.4f}'.format(eta2/eta1))
        # plt.colorbar(im3)
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_xlim([-18,18])
        # ax.set_ylim([-18,24])
        # if (saveFigures):
        #     plt.savefig(path_figures + H + "/" + H + '_LevelSetsExact_' + str(k) + '.png', dpi=my_dpi * 10)
        # # Level sets and gradients
        # fig = plt.figure(figsize=(800/my_dpi, 800/my_dpi), dpi=my_dpi)
        # im4 = plt.contourf(xi, yi, zi_lin, cmap = colormap2, levels = 30 )
        # plt.quiver(xiG, yiG, trueGMesh[:, 0], trueGMesh[:, 1])
        # plt.title("Level sets exact solution and gradients, ratio=" + '{:.4f}'.format(eta2/eta1))
        # plt.colorbar(im4)
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_xlim([-18,18])
        # ax.set_ylim([-18,24])
        # if (saveFigures):
        #     plt.savefig(path_figures + H + "/" + H + '_LevelSetsExactGrads_' + str(k) + '.png', dpi=my_dpi * 10)
        # # Type of solution
        # fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
        # im5 = plt.scatter(eik_coords[:, 0], eik_coords[:, 1], s = 2 + round(7000/len(eik_coords)), c = typeSol, cmap = colormap3)
        # plt.colorbar(im5)
        # plt.title("Type of solution, ratio=" + '{:.4f}'.format(eta2/eta1))
        # ax = plt.gca()
        # ax.set_aspect('equal')
        # ax.set_xlim([-18,18])
        # ax.set_ylim([-18,24])
        # if (saveFigures):
        #     plt.savefig(path_figures + H + "/" + H + '_TypeSol_' + str(k) + '.png', dpi=my_dpi * 10)



def generateAnimation(k, name = "LevelSetsExact", pathSave = path_figures):
    '''
    Using the matplotlib animation library we read the images computed by
    solving the eikonal equation in a domain + changing the slowness function
    '''
    ims = []
    fig = plt.figure(figsize = (800/my_dpi, 800/my_dpi), dpi = my_dpi)
    for i in range(k):
        if(i == 50):
            continue
        # Read
        fname = path_figures + H + "/" + H + "_" + name + "_" + str(i) + ".png"
        im_i = plt.imread(fname)
        ims.append( [plt.imshow(im_i, cmap=colormap2, animated=True)])
    #breakpoint()
    ani = animation.ArtistAnimation(fig, ims, interval = 50, blit = True, repeat_delay = 1000)
    ani.save(pathSave + H + "/" + H + "_ANIMATED" + name + ".gif")



def saveSnapshots(J, name, pathSaved = path_figures, pathSaveTo = path_figures):
    '''
    Given J files in with the name format in the pathSaved path
    we create the necessary file to store them in snapshot format.
    '''
    print("Trying to organize snapshots for " + name)
    assert(J > 0, "We need more than 1 snapshot")
    # Open the first file to see the dimensions we need
    fname = pathSaved + H + "/" + H + "_" + name + "_0.txt"
    data = np.genfromtxt(fname, delimiter =",")
    data_flat = data.reshape(-1)
    N = len(data_flat)
    snapshots = np.empty((N, J-1))
    print("Snapshot matrix of size " + str(N) + " x " + str(J-1))
    snapshots[:, 0] = data_flat
    # Start reading
    print("Starting to read documents")
    for i in range(1, J-1):
        if(i == 50):
            continue
        # Read
        fname = pathSaved + H + "/" + H + "_" + name + "_" + str(i) + ".txt"
        data = np.genfromtxt(fname, delimiter = ",")
        data_flat = data.reshape(-1)
        snapshots[:, i] = data_flat
    # Then we save
    print("Finish reading, now saving")
    if( name == "typeSol"):
        np.savetxt(pathSaveTo + H + "_snapshots_" + name + ".txt", snapshots, delimiter = ", ", fmt = "%.0f")
    else:
        np.savetxt(pathSaveTo + H + "_snapshots_" + name + ".txt", snapshots, delimiter = ", ", fmt = "%.12f")
    
    
    
    
#generateAnimation(79)    
            
