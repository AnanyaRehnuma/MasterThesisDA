import numpy as np
from scipy.linalg import eig, eigh
from scipy.stats import norm
import scipy.stats as ss
import pandas as pd
from plotly import express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from multidomain_MMD import L, H
import matplotlib.pyplot as plt


class uKSE:
    """Univariate squared exponential kernel."""
    def __init__(self, h):
        """Bandwidth parameter will be squared."""
        self.h = h
    def __call__(self, x, y):
        return np.exp(-0.5*(x-y)**2/self.h**2)/np.sqrt(2*np.pi)/self.h


def MMD(x, y, k: callable) -> float:
    """Univariate two-sample MMD.

    Args:
        x, y: The two samples; flat array
        k: Kernel function

    Returns: Calculated MMD
    """
    kxx = k(x[None,:], x[:,None])
    kxy = k(x[None,:], y[:,None])
    kyy = k(y[None,:], y[:,None])
    return kxx.sum()/x.size**2 - 2*kxy.sum()/x.size/y.size + kyy.sum()/y.size**2


class MixtureModel(ss.rv_continuous):
    def __init__(self, submodels, *args, weights = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        if weights is None:
            weights = [1 for _ in submodels]
        if len(weights) != len(submodels):
            raise(ValueError(f'There are {len(submodels)} submodels and {len(weights)} weights, but they must be equal.'))
        self.weights = [w / sum(weights) for w in weights]
        
    def _pdf(self, x):
        pdf = self.submodels[0].pdf(x) * self.weights[0]
        for submodel, weight in zip(self.submodels[1:], self.weights[1:]):
            pdf += submodel.pdf(x)  * weight
        return pdf

    def rvs(self, size):
        submodel_choices = np.random.choice(len(self.submodels), size=size, p = self.weights)
        submodel_samples = [submodel.rvs(size=size) for submodel in self.submodels]
        rvs = np.choose(submodel_choices, submodel_samples)
        return rvs
    
if __name__ == '__main__':
    import scipy.stats as ss

    mixture_model_x = MixtureModel([ss.norm(-3, 1), 
                              ss.norm(3, 1), 
                              ss.uniform(loc=3, scale = 2)],
                             weights = [0.1, 0.4, 0.2])
    
    mixture_model_y = MixtureModel([ss.norm(-1, 1), 
                              ss.norm(10, 1), 
                              ss.uniform(loc=3, scale = 1)],
                             weights = [0.2, 0.3, 0.2])
    
    x_samples = mixture_model_x.rvs(100)
    y_samples = mixture_model_y.rvs(100)

    x_axis = np.arange(-6, 6, 0.1)
    y_axis = np.arange(-3, 11, 0.1)
    
    # See what how the MMD looks if parameters of Y are altered
    if False:
        mus, sds = np.ogrid[-2:2:50j, 00.1:5:50j]
        it = np.nditer([mus, sds, None])
        for mu, sd, mmd in it:
            y_samples = np.random.normal(loc=mu, scale=sd, size=150)
            xy = np.concatenate([x_samples, y_samples])
            N = xy.size

            sigma2 = N**(-2/(1+4))
            sigma2 *= np.var(xy)
            k = uKSE(np.sqrt(sigma2))
            K = k(xy[:,None], xy[None,:])
            L_ = L(x_samples.size, y_samples.size)
            mmd[...] = N**2*np.trace(K@L_)

        fig = px.imshow(np.log(it.operands[-1]), y=mus.flatten(), x=sds.flatten())
        fig.add_scatter(x=[1], y=[0])
        fig.show()

    xy = np.concatenate([x_samples, y_samples])
    N = xy.size
    n_x = len(x_samples)
    n_y = len(y_samples)

    # Bandwidth parameters; Scott's rule of thumb
    sigma2s = np.power([n_x, n_y], -2./(1+4))
    sigma2s *= np.var(x_samples), np.var(y_samples) #empiric variance from samples
    k = uKSE(np.sqrt(sigma2s.mean()))
    # Kernel matrix
    K_xyxy = k(xy[:,None], xy[None,:])
    # Coefficient Matrix
    L_ = L(x_samples.size, y_samples.size)
    # Centering matrix
    H_ = H(N)

    # # eigen decomposition
    # KHK = K_xyxy@H_@K_xyxy
    # KLK = K_xyxy@L_@K_xyxy
    # mu = 0.01*np.abs(KLK).max()
    # KLK += mu*np.identity(N)
    # # Solve KHK @ vecs = (KLK + mu*I) @ vecs @ vals
    # vals, vecs = eigh(KHK, KLK, subset_by_index=[N-100, N-1])


    # Solve the INVERSE generalized eigenvalue problem 
    KLK = K_xyxy@L_@K_xyxy
    # Add identity to regularize KHK for being positive definite
    # XXX Regularizes the KHK other than in the paper 
    KHK = K_xyxy@H_@K_xyxy    
    while True:
        try: 
            np.linalg.cholesky(KHK)
        except np.linalg.linalg.LinAlgError:
            KHK += 1e-6*np.identity(N)
        else:
            break

    # Dimensionality of the latent space i.e. leading eigenvectors 
    m = 5
    # solve the 'reverse' gEVP
    vals, vecs = eigh(KLK, KHK, subset_by_index=[N-m, N-1])
    # Double-check if vecs and vals solve the gEVP 
    assert np.allclose(KLK@vecs, KHK @ vecs @ np.diag(vals)), 'Does not solve the gEVP.'
    # Double-check if the secondary constraint is fulfilled 
    # assert np.allclose(vecs.T@KHK@vecs, np.identity(m), atol=1e-6), 'Constraint W^TKHKW = I not fulfilled!'

    # KDE using the proxy kernel 
    z, dz = np.linspace(-20, 20, 400, retstep=True)
    K_zxy = k(z[:,None], xy[None,:])
    K_tilde = K_zxy@vecs@vecs.T@K_xyxy

    if False:
        fig = go.Figure(layout=dict(barmode='overlay'))
        for i, (mu, sd, X) in enumerate([(x_samples.mean(), x_samples.std(), x_samples), (y_samples.mean(), y_samples.std(), y_samples)], start=1):
            fig.add_scatter(x=z, y=norm.pdf(z, loc=mu, scale=sd), name='pdf', legendgroup=i, line=dict(color=DEFAULT_PLOTLY_COLORS[i-1]))
            fig.add_histogram(x=x_samples, showlegend=False, histnorm='probability', opacity=0.75)
            kde = k(z[:,None], X[None,:]).sum(axis=-1)/len(X)
            fig.add_scatter(x=z, y=kde, name='kde', legendgroup=i, line=dict(color=DEFAULT_PLOTLY_COLORS[i-1]))

        kde_tilde = K_tilde.sum(axis=-1)
        kde_tilde /= kde_tilde.sum()*dz
        fig.add_scatter(x=z, y=kde_tilde, name='TCA')

        fig.show()
        exit()

    if False:
        x_new = vecs[:n_x,:].T@K_xyxy[:n_x,:n_x]
        y_new = vecs[n_x:,:].T@K_xyxy[n_x:,n_x:]
        fig = go.Figure()
        fig.add_scatter(x=x_new[0,:], y=x_new[1,:], mode='markers')
        fig.add_scatter(x=y_new[0,:], y=y_new[1,:], mode='markers')
        fig.show()
        exit()


    if True: 
        # plt.plot(x_axis, mixture_model_x.pdf(x_axis), label = 'PDF')
        # plt.plot(y_axis, mixture_model_y.pdf(y_axis), label = 'PDF')

        plt.hist(x_samples, bins = 50, density = True, label = 'x_Sampled', alpha = 0.9)
        plt.hist(y_samples, bins = 50, density = True, label = 'y_Sampled', alpha = 0.7)
        kde_xnorm = k(z[:,None], x_samples[None,:]).sum(axis=-1)/len(x_samples)
        plt.plot(z, kde_xnorm, label = 'kde_x_samples')
        kde_ynorm = k(z[:,None], y_samples[None,:]).sum(axis=-1)/len(y_samples)
        plt.plot(z, kde_ynorm, label = 'kde_y_samples')
        kde_tilde = K_tilde.sum(axis=-1)
        kde_tilde /= kde_tilde.sum()*dz
        plt.plot(z, kde_tilde, label = 'tca')
        plt.legend()

        plt.show()
