import numpy as np

from skimage import io

from math import sqrt


def broadcast_sqr(X):
    tslice = np.transpose(X)
    A = np.rot90(tslice[:, :, None], 1, (1, 2))
    B = np.rot90(A, 1, (0, 1))
    # A.shape[0] == B.shape[1]
    # A.shape[2] == B.shape[2]
    C = np.empty((A.shape[2], A.shape[0], B.shape[1]))
    for j, m in zip(range(A.shape[2]), range(B.shape[2])):
        C[j, :, :] = A[:, :, j] * B[:, :, m]

    return C


def get3DCovVar(array):
    mean = np.mean(array, axis=0)
    delta = array - mean
    cov = []
    for i in range(delta.shape[1]):
        tslice = np.transpose(delta[:, i, :])
        A = np.rot90(tslice[:, :, None], 1, (1, 2))
        B = np.rot90(A, 1, (0, 1))
        cov.append(np.sum(A * B, axis=2))
    cov = np.array(cov) / len(array)
    return cov, mean


def get3DCovVar2(array):
    mean = np.mean(array, axis=0)
    delta = array - mean
    cov = []
    for i in range(delta.shape[0]):
        tslice = np.transpose(delta[i, :, :])

        A = np.rot90(tslice[:, :, None], 1, (1, 2))
        B = np.rot90(A, 1, (0, 1))
        cov.append(A * B)

    cov = np.sum(cov, axis=0) / len(array)
    return cov, mean

def get3DCovVar3(array):
    mean = np.mean(array, axis=0)
    delta = array - mean
    cov = []
    for i in range(delta.shape[0]):
        print("\r", i,"           ", end="")
        cov.append(broadcast_sqr(delta[i, :, :]))
#this is very slow
    cov = np.sum(cov, axis=0) / len(array)
    return cov, mean


def get3DCov(array):
    mean = np.mean(array, axis=0)
    delta = array - mean
    cov = []
    for i in range(delta.shape[1]):
        slice = delta[:, i, :]
        cov.append(np.matmul(np.transpose(slice), slice))
    cov = np.array(cov) / len(array)
    return cov, mean


def get2DCov(array):
    mean = np.mean(array, axis=0)
    delta = array - mean
    cov = []
    for i in range(delta.shape[0]):
        slice = delta[i, :]
        cov.append(np.matmul(slice.reshape((slice.shape[0], 1)), slice.reshape((1, slice.shape[0]))))
    cov = np.sum(cov, axis=0) / len(array)
    return cov, mean


# array.shape[0] > array.shape[1] иначе ковариационная матрица получится вырожденной
def generate_ascent(array):
    print("Optimizer calculating...")
    cov, mean = get3DCov(array)

    print(cov.shape)
    cholM = []
    for i in range(cov.shape[0]):
        cholM.append(np.linalg.cholesky(cov[i]))
    cholM = np.array(cholM)
    print(cholM.shape)
    cholML = np.empty(cholM.shape)
    cholMR = np.empty(cholM.shape)
    for i in range(cholM.shape[1]):
        cholML[:, i, :] = cholM[:, i, :] - mean
        cholMR[:, i, :] = cholM[:, i, :] + mean
    cholM = np.concatenate((cholML, cholMR), axis=1)

    cholM = cholM.swapaxes(0, 1)
    return cholM


dets = []
def generate_ascent2D(array):
    cov, mean = get2DCov(array)
    det = np.linalg.slogdet(cov)
    if det[0] == -1:
        print("Negative!!")
    if det[0] == 0:
        print("Zero!!")
    dets.append(det[1])

    cholM = np.linalg.cholesky(cov)
    cholML = np.empty(cholM.shape)
    cholMR = np.empty(cholM.shape)

    for i in range(cholM.shape[1]):
        cholML[:, i] = cholM[:, i] - mean
        cholMR[:, i] = cholM[:, i] + mean
    # cholM = np.concatenate((cholML, cholMR), axis=1)
    cholM = cholML + cholMR
    cholM = cholM.swapaxes(0, 1)
    return cholM
