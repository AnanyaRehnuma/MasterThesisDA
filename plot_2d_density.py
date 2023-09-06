from itertools import pairwise, accumulate
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from multidomain_MMD import KSE as mKSE
from multidomain_MMD import H, L
from scipy.linalg import eigh

def make_fig():
    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                        left=0.15, right=0.98, bottom=0.1, top=1,
                        wspace=0.05, hspace=0.05)

    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.set_ylabel(ac_col)
    ax.set_xlabel(cot_col)
    return fig, ax, ax_histx, ax_histy

dr = 0.5  # Discretization of the tunnel-distance 

cot_col = 'cot_ver'
ac_col = 'ac_angle_ver'

tunnels = pd.read_excel('list.xlsx', index_col='thesis_name')
tunnels.query("'TBM A' == thesis_name" or "'TBM B' == thesis_name", inplace=True)

### Read all source domain data ###
ac_all = [] # list to collect all ac data
cot_all = [] # list to collect all cot data
ns = [] # list to collect the lengths of the source domains
hist_cot_max = 0
hist_ac_max = 0
for tunnel_name, row in tunnels.iterrows():
    path = row['path'].rsplit('\\', maxsplit=1)[0] 
    file = f'{path}\\preprocessed_{dr}.csv'
    data = pd.read_csv(file).dropna()
    ac_all.append(data[ac_col])
    cot_all.append(data[cot_col])
    counts, edges = np.histogram(data[cot_col], density=True, bins='scott')
    hist_cot_max = max(hist_cot_max, max(counts))
    counts, edges = np.histogram(data[ac_col], density=True, bins='scott')
    hist_ac_max = max(hist_ac_max, max(counts))
    ns.append(len(data))

# Total number of records
N = sum(ns)

# Concatenate data 
cot_all = pd.concat(cot_all, keys=tunnels.index, names=['name', 'distance'])
ac_all = pd.concat(ac_all, keys=tunnels.index, names=['name', 'distance'])

if True:
    # Estimate kernel bandwidth 
    # FIXME: Find a better way to estimate the bandwidth matrix :
    Sigma = np.cov([cot_all, ac_all]) # Empiric Covariance
    Sigma *= len(data)**(-2/(2+4)) # Scale by Scott's rule 

    # Instantiate Gaussian kernel 
    k = mKSE(Sigma)
    # Evaluate kernel Matrix
    src = np.column_stack([cot_all, ac_all])
    K = k(src[:,None,:], src[None,:,:])

    # Centering Matrix
    H = H(N)
    # Coefficient Matrix
    L = L(*ns)

    KLK = K@L@K
    KHK = K@H@K
    i = -6  # Start to regularize from 10^-6 on 
    while True:
        try: 
            # Try to calculate eigenvalues and eigenvectors 
            vals, vecs = eigh(KLK, KHK)
        except np.linalg.linalg.LinAlgError:
            KHK += 10**i*np.identity(N)  # Regularize by adding identity 
            print(f'added {10**i} of identity to regularize KHK')
        else:
            break
        i += 1 # Increase the order of magnitude 

    # Eigenvalues may be negative because both KLK and KHK are not necessarily positive definite 
    # Sort eigenvectors by absolute value
    idx = np.argsort(np.abs(vals))
    # idx sorts the eigenvectors by importance 
    vecs = vecs[:,idx]

    # Dimensionality of the latent space i.e. leading eigenvectors 
    # vecs is the column-wise eigenvector matrix 
    m = 6 # XXX Some reasoning for choosing m were good 
    # Keep the trailing m eigenvectors 
    # an eigenvector has length N i.e. all records 
    vecs = vecs[:,-m:]

    # Calculate transferred kernel 
    # WWTK = vecs@vecs.T@K
    # K_tilde = K@WWTK


# Span a grid 
cot_ptp = np.ptp(cot_all) # Point-to-point 
exp = 0.15
cot_min, cot_max = cot_all.min()-exp*cot_ptp, cot_all.max()+exp*cot_ptp
ac_ptp = np.ptp(ac_all)
ac_min, ac_max = ac_all.min()-exp*ac_ptp, ac_all.max()+exp*ac_ptp
cots, acs = np.mgrid[cot_min:cot_max:250j, ac_min:ac_max:251j]
dcot = cots[1,0] - cots[0,0]  # Discretization CoTs
dac = acs[0,1] - acs[0,0]  # Discretization ACs

ys = np.stack([cots, acs], axis=-1)  # Where do evaluate the density estimate 
K_zxy = k(ys[:,:,None,:], src[None, None,:,:])
k_cot_ac_tilde = K_zxy@vecs@vecs.T@K
# compute the kde 
kde_cot_ac_tilde = k_cot_ac_tilde.sum(-1)
# normalize such that the density sums to unity 
kde_cot_ac_tilde /= kde_cot_ac_tilde.sum()*dcot*dac

# Loop over tunnels an save plots 
for tunnel_name, row in tunnels.iterrows():
    path = row['raw_data_path'].rsplit('\\', maxsplit=1)[0] 
    file = f'{path}\\preprocessed_{dr}.csv'
    data = pd.read_csv(file).dropna()

    # Estimate bandwidth 
    # XXX Which bandwidth estimate do we use
    # Sigma = len(data)**(-2/(2+4))*np.cov([cot, ac])
    # k = mKSE(Sigma)  # Instantiate Gaussian kernel 

    xy = data[[cot_col,ac_col]].values
    # Evaluate kernel 
    K_zxy = k(ys[:,:,None,:],xy[None, None,:,:])
    # Density estimate
    kde_cot_ac = K_zxy.sum(-1)/len(data)

    # Integrate out CoT 
    kde_ac = kde_cot_ac.sum(axis=0)*dcot
    kde_ac_tilde = kde_cot_ac_tilde.sum(axis=0)*dcot
    # Integrate out AC
    kde_cot = kde_cot_ac.sum(axis=1)*dac
    kde_cot_tilde = kde_cot_ac_tilde.sum(axis=1)*dac

    ### Figure with marginals right and atop ###
    fig, ax, ax_histx, ax_histy = make_fig()

    # CoT histogram 
    counts, bins_cot, _ = ax_histx.hist(data[cot_col], density=True, bins='scott')
    # Density estimate 
    ax_histx.plot(cots[:,0], kde_cot, color='orange')
    ax_histx.plot(cots[:,0], kde_cot_tilde, color='green')
    ax_histx.set_xlim(cot_min, cot_max)
    ax_histx.set_ylim(0, hist_cot_max)

    # AC histogram
    counts, bins_aca, _ = ax_histy.hist(data[ac_col], density=True, orientation='horizontal', bins='scott')
    # Density estimate 
    ax_histy.plot(kde_ac, acs[0,:], color='orange')
    ax_histy.plot(kde_ac_tilde, acs[0,:], color='green')
    ax_histy.set_ylim(ac_min, ac_max)
    ax_histy.set_xlim(0, hist_ac_max)


    # Scatter plot 
    ax.scatter(data[cot_col], data[ac_col], marker='.')
    # kde contour plot 
    ax.contour(cots, acs, kde_cot_ac, levels=10, colors='orange')
    ax.contour(cots, acs, kde_cot_ac_tilde, levels=10, colors='green')

    fig.savefig(f'{path}\\marginals.png')

    ### With cross sections right and atop ###
    fig, ax, ax_histx, ax_histy = make_fig()

    counts, bins_cot, bins_aca, _ = ax.hist2d(data[cot_col], data[ac_col], bins=(len(bins_cot), len(bins_aca)), density=True, cmap='hot_r', vmin=0)
    center_cot = (bins_cot[1:] + bins_cot[:-1])/2
    center_aca = (bins_aca[1:] + bins_aca[:-1])/2
    idx_cot, idx_aca = np.unravel_index(counts.argmax(), counts.shape)

    # kde contour plot 
    ax.contour(cots, acs, kde_cot_ac, levels=10, colors='orange')
    ax.contour(cots, acs, kde_cot_ac_tilde, levels=10, colors='green')

    ax.hlines(center_aca[idx_aca], cot_min, cot_max)
    ax.vlines(center_cot[idx_cot], ac_min, ac_max)
    ax.set_xlim(cot_min, cot_max)
    ax.set_ylim(ac_min, ac_max)


    ax_histx.bar(center_cot, counts[:,idx_aca], center_cot.ptp()/(counts.shape[0]-1))
    ax_histx.set_xlim(cot_min, cot_max)

    ax_histy.barh(center_aca, counts[idx_cot,:], center_aca.ptp()/(counts.shape[1]-1))
    ax_histy.set_ylim(ac_min, ac_max)

    idx_aca = np.abs(acs[0,:] - center_aca[idx_aca]).argmin()
    ax_histx.plot(cots[:,0], kde_cot_ac[:,idx_aca], color='orange')
    ax_histx.plot(cots[:,0], kde_cot_ac_tilde[:,idx_aca], color='green')

    idx_cot = np.abs(cots[:,0] - center_cot[idx_cot]).argmin()
    ax_histy.plot(kde_cot_ac[idx_cot,:], acs[0,:], color='orange')
    ax_histy.plot(kde_cot_ac_tilde[idx_cot,:], acs[0,:], color='green')

    fig.savefig(f'{path}\\slices.png')