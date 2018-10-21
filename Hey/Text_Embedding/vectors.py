import numpy as np

arr = np.loadtxt("xaa")
np.save("arr1.npy",arr)
arr = np.loadtxt("xab")
np.save("arr2.npy",arr)
arr = np.loadtxt("xac")
np.save("arr3.npy",arr)
arr = np.loadtxt("xad")
np.save("arr4.npy",arr)
