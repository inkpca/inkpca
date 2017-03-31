
# -*- coding: utf-8 -*-

from __future__ import division, print_function

# This package
import data
from incremental_kpca import IncrKPCA, nystrom_approximation
from kernels import kernel_matrix, rbf, adjust_K, median_distance

# Built-in modules
import sys

# External modules
import numpy as np
from numpy import dot, diag
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Matplotlib
rcParams['font.family'] = 'serif'
rcParams['axes.titlesize'] = 21
rcParams['axes.labelsize'] = 19
rcParams['xtick.labelsize'] = 15
rcParams['ytick.labelsize'] = 15


def main(dataset='magic', datasize=1000):
    """
    Run the experiments for our incremental algorithms
    """

    if not dataset in ('magic', 'yeast'):
        raise ValueError("Unknown dataset.")

    # Data
    get_data_fcn = getattr(data, "get_" + dataset + "_data")
    X = get_data_fcn()
    if datasize:
        Xcut = X[:datasize] # Smaller dataset for Nyström comparison

    sigma = median_distance(X)

    kernel = lambda x, y: rbf(x, y, sigma)

    mmax = 100

    m0 = 20

    incremental_experiment(X, m0, mmax, kernel, dataset)

    incremental_experiment_mean_adjusted(X, m0, mmax, kernel, dataset)

    nystrom_experiment(Xcut, m0, mmax, kernel, dataset)

    nystrom_experiment_mean_adjusted(Xcut, m0, mmax, kernel, dataset)


def incremental_experiment(X, m0, mmax, kernel, dataset):
    """
    Experiment of incremental kernel pca algorithm
    """
    print("KPCA")
    inc = IncrKPCA(X, m0, mmax, kernel=kernel)
    fnorms = []
    for i, L, U in inc:
        idx = inc.get_idx_array()
        K = kernel_matrix(X, kernel, idx[:i+1], idx[:i+1])
        K_tilde = dot(U, dot(diag(L), U.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(range(m0,mmax), fnorms, dataset, "m", "Frobenius norm")

def incremental_experiment_mean_adjusted(X, m0, mmax, kernel, dataset):
    """
    Experiment of incremental kernel pca algorithm with mean adjustment
    """
    print("KPCA mean-adjusted")
    inc = IncrKPCA(X, m0, mmax, adjust=True, kernel=kernel)
    fnorms = []
    for i, L, U in inc:
        idx = inc.get_idx_array()
        K = kernel_matrix(X, kernel, idx[:i+1], idx[:i+1])
        K = adjust_K(K)
        K_tilde = dot(U, dot(diag(L), U.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(np.arange(len(fnorms))+m0, fnorms, dataset, "m", "Frobenius norm")

def nystrom_experiment(X, m0, mmax, kernel, dataset):
    """
    Incremental kernel pca algorithm with the Nyström method. Plots the
    difference between the kernel matrix and its approximation for different
    size subsets.
    """
    print("Nystrom KPCA")

    inc = IncrKPCA(X, m0, mmax, kernel=kernel, nystrom=True)#, r=10)
    idx = inc.get_idx_array()
    n = X.shape[0]
    K = kernel_matrix(X, kernel, range(n), range(n))
    fnorms = []
    for i, L, U, L_nys, U_nys in inc:
        K_tilde = dot(U_nys, dot(diag(L_nys), U_nys.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(K - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(range(m0, m0+len(fnorms)), fnorms, dataset, "m", "Frobenius norm")

def nystrom_experiment_mean_adjusted(X, m0, mmax, kernel, dataset):
    """
    Incremental kernel pca algorithm with Nyström method, when adjusting the
    mean of feature vectors.
    """
    print("Nystrom KPCA mean-adjusted")

    inc = IncrKPCA(X, m0, mmax, kernel=kernel, nystrom=True, adjust=True, r=10)
    n = X.shape[0]
    K = kernel_matrix(X, kernel, range(n), range(n))
    Kp = adjust_K(K)
    fnorms = []
    for i, L, U, L_nys, U_nys in inc:
        K_tilde = dot(U_nys, dot(diag(L_nys), U_nys.T))
        fnorm = np.sqrt(np.sum(np.sum(np.power(Kp - K_tilde, 2))))
        fnorms.append(fnorm)

    plotting(range(m0, m0+len(fnorms)), fnorms, dataset, "m", "Frobenius norm")


#################
##### UTILS #####
#################

def plotting(x, y, title, xlabel, ylabel):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()

#################


if __name__ == '__main__':
    main(*sys.argv[1:])
