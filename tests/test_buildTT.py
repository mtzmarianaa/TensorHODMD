import pytest
import numpy as np
from src.tutils import *
import pickle



# Read all the data and store it in a tensor, then
# convert this to a tensor train


path_info = "tests/TestEikonal/TestEikonal/"

# Notice, here we read ALL the information, after this
# we would need to separate the respective X and Y tensors
# to do DMD
n1 = 10 #x0 position
n2 = 68150 # points
n3 = 4 # eikonal, gradient, type of solution
m = 40


TX = np.empty((n1, n2, n3, m)) #x0 position, points, eikonal, gradient, type of solution

for k in range(m):
    for i1 in range(n1):
        H = "H" + str(i1)
        path_info = "tests/TestEikonal/TestEikonal/" + H + "/" + H + "_"
        eik = np.genfromtxt(path_info + "true_values_" + str(k) + ".txt", delimiter = ",")
        grads = np.genfromtxt(path_info + "true_grads_ " + str(k) + ".txt", delimiter = ",")
        typeSol = np.genfromtxt(path_info + "typeSol_" + str(k) + ".txt", delimiter = ",")
        TX[i1, :, 0, k] = eik
        TX[i1, :, 1, k] = grads[:, 0]
        TX[i1, :, 2, k] = grads[:, 1]
        TX[i1, :, 3, k] = typeSol

# Save this to disc using pickle
 file = "tests/TX"
 pickle.dump(TX, file)
 file.close()


