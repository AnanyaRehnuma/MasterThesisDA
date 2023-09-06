import numpy as np
from numpy.typing import NDArray
from itertools import pairwise, accumulate

def Lii(ni: int, S: int, N: int) -> NDArray:
    return np.full([ni, ni], (S-1)/N**2/ni**2)

def Lij(ni: int, nj: int, N: int) -> NDArray:
    return np.full([ni, nj], -1/N**2/ni/nj)

def L(*ns) -> NDArray:
    """Calculates the coefficient matrix for multi-domain MMD

    Test case for the diagonal 
    >>> S = 3
    >>> n1=3; n2=2; n3=4
    >>> N = n1 + n2 + n3
    >>> diag = np.concatenate([np.full(n1, (S-1)/(N*n1)**2), 
    ...                        np.full(n2, (S-1)/(N*n2)**2), 
    ...                        np.full(n3, (S-1)/(N*n3)**2)])
    >>> L3 = L(n1,n2,n3)
    >>> assert np.allclose(L3.diagonal(), diag)

    Check off-diagonal blocks 
    >>> L2 = L(1,1)
    >>> assert L2[0,1] == -1/4
    >>> assert L2[0,1] == L2[1,0]
    """
    S = len(ns) # Number of source domains
    cum_ns = tuple(accumulate(ns, initial=0)) # Add zero to start counting from
    N = cum_ns[-1] # Total number of records
    L = np.empty([N, N]) # Allocate memory 
    for i, (start_i, stop_i) in enumerate(pairwise(cum_ns)):
        ni = ns[i] # Number of records of the i-th column 
        slc_i = slice(start_i, stop_i) # Slice for the i-th data set 
        L[slc_i, slc_i] = Lii(S=S, ni=ni, N=N)
        for j, (start_j, stop_j) in enumerate(pairwise(cum_ns[i+1:]), start=i+1):
            nj = ns[j]
            slc_j = slice(start_j, stop_j)
            L[slc_i, slc_j] = Lij(ni=ni, nj=nj, N=N)
            L[slc_j, slc_i] = L[slc_i, slc_j].T
    return L


class KSE:
    def __init__(self, Sigma):
        self.Sigma = np.asarray(Sigma)
        self.det = np.linalg.det(Sigma)
        self.n = np.sqrt((2*np.pi)**2 * self.det)

    def __call__(self, x, y):
        #return np.exp(-2*(x-y))**2/self.Sigma**2
        # FIXME: Normalization is missing 
        return (np.exp(-0.5*np.einsum('...i, ij, ...j -> ...', np.subtract(x,y), np.linalg.inv(self.Sigma), np.subtract(x,y)))/self.n)


def kde(k, *xs) -> NDArray:
    x_new = np.vstack(xs)
    return k(x_new[:,None,:], x_new[None,:,:])


def H(N:int) -> NDArray:
    """Calculating the centering matrix

    Check if the matrix is symmetric
    >>> N = 15
    >>> arr = H(N)
    >>> assert np.allclose(arr.transpose(),arr)

    Check if the matrix is idempotent; when multiplied by itself yields itself
    >>> N = 10
    >>> arr = H(N)
    >>> assert np.allclose(arr@arr, arr)

    Check tr(H) = n - 1
    >>> assert np.allclose(np.trace(H(11)), 10)
    """
    return np.identity(N) - np.full([N,N], 1/N)


if __name__ == '__main__':

    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.linalg import eig, eigh

    df1 = pd.read_csv('./thesis_code/data/s001180_1/preprocessed_0.5.csv').head(500)
    df2 = pd.read_csv('./thesis_code/data/s001180_2/preprocessed_0.5.csv').head(500)
    df3 = pd.read_csv('./thesis_code/data/s001180_3/preprocessed_0.5.csv').head(500)

    x1 = df1[['cot_ver','ac_angle_ver']].values
    x2 = df2[['cot_ver','ac_angle_ver']].values
    x3 = df3[['cot_ver','ac_angle_ver']].values

#     # n1 = len(df1)

#     # bins, edges_cot, edges_aca = np.histogram2d(x=x1[:,0], y=x1[:,1], bins=[20,21], density=True)

#     # idx_cot, idx_aca = np.unravel_index(np.argmax(bins), bins.shape)

#     # center_aca = (edges_aca[1:] + edges_aca[:-1])/2
#     # center_cot = (edges_cot[1:] + edges_cot[:-1])/2

#     # m = 2 # Dimension is 1
#     # Sigma = n1**(-2/(m+4)) 
#     # Sigma *= df1[['cot_ver', 'ver']].cov() 
#     # k = KSE(Sigma)


#     # # TODO: Use fine sampling to draw nice curves 
#     # y_aca = np.column_stack([np.full_like(center_aca, center_cot[idx_cot]),center_aca])
#     # KD_cent_aca = k(y_aca[:,None,:], x1[None,:,:]).sum(axis=-1)/n1
#     # plt.plot(center_aca, bins[idx_cot,:]) # TODO: plot histogram 
#     # plt.plot(center_aca, KD_cent_aca)
#     # plt.show()

#     # # TODO: Calculate the KDE evaluated at (center_cot, center_aca[idx_aca])
#     # cots = np.linspace(edges_cot.min(), edges_cot.max(), 201)
#     # y_cot = np.column_stack([cots, np.full_like(cots, center_aca[idx_aca])])
#     # KD_cent_cot = k(y_cot[:,None,:], x1[None,:,:]).sum(axis=-1)/n1
#     # plt.plot(center_cot, bins[:, idx_aca])
#     # plt.plot(cots, KD_cent_cot)
#     # plt.show()

#     # exit()

    m = 2 # Dimension is 2
    S = 3
    n1 = len(x1)
    n2 = len(x2)
    n3 = len(x3)
    
    N = n1+n2+n3
    print(N)
    Sigma = n1**(-2/(m+4)) 
    # TODO: Calculate cov from x123
    Sigma *= df1[['cot_ver', 'ac_angle_ver']].cov() 
    k = KSE(Sigma)

    K = kde(k, x1, x2, x3)
    L_ = L(n1, n2, n3)
    H_ = H(N)
    #multiMMD = (K@L_).trace()
    #print(multiMMD)
    # formulae (KLK + μI)−1KHK 
    mu = 0.001

    KLK = K@L_@K
    KHK = K@H_@K
    Lambda, v = eigh(KHK, KLK + mu*np.identity(N))
    print(KHK@v)
    print((KLK + mu*np.identity(N)) @ v @ np.diag(Lambda))
    #assert np.allclose(KHK@v, (KLK + mu*np.identity(N)) @ v @ np.diag(Lambda))

    # # now we reverse engineer the to get the same r matrix back from the eigenvalues Lambda and eigen vector v
    p = np.linalg.inv(KLK + mu*np.identity(n=N))
    r = p@KHK # XXX not symmetric anymore?
    vecs_inv = np.linalg.inv(v)
    r_back = v @ np.diag(Lambda) @ vecs_inv
    assert np.allclose(r_back, r)
    exit()
    vec = v @ np.diag(Lambda) @ np.linalg.inv(v)
    M = v.T @ np.diag(Lambda) @ v
    #print(np.diag(Lambda))
    L_tilde = Lambda
    L_tilde[-1] = 0
    print(M)
    M_tilde = v.T @ np.diag(L_tilde) @ v
    print(M_tilde)
    #print(r.nbytes/1024/1024)
    #assert np.allclose(vec, M)
    print(np.abs(M_tilde - M))

    exit()