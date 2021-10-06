import numpy as np
import cvxpy as cp
from sklearn.metrics.pairwise import rbf_kernel as sklearn_kernel
from sklearn.metrics import euclidean_distances
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH
import matplotlib.pyplot as plt


def cholesky_decomposition(K):
    K = (K.T + K) / 2 + np.eye(K.shape[0]) * 10e-5
    assert np.all(K.T == K), "Kernel matrix is not symmetric"
    L = np.linalg.cholesky(K)
    return L


def rkhs_dist(x, wx, y, wy, kernel, **param):
    # compute gram matrix
    Kxx, Kxxp, Kxpxp = gram_mat(x, y, kernel=kernel, **param)

    # compute rkhs norm squared
    cost_rkhs = wy.T @ Kxpxp @ wy - 2 * (
            wx.T @ Kxxp @ wy) + wx.T @ Kxx @ wx

    return np.sqrt(cost_rkhs) # return squared norm


def gram_mat(x, y, kernel=None, **param):
    '''compute gram matrix'''
    if kernel is None:
        kernel = rbf_kernel

    if len(np.asarray(x).shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(np.asarray(y).shape) < 2:
        y = np.expand_dims(y, axis=1)

    # first
    K11 = kernel(x,x, **param)

    # second
    K22 = kernel(y,y, **param)

    # cross term
    K12 = kernel(x,y, **param)
    return K11, K12, K22


def compute_mmd(X, Y, sigma):
    """
    Compute MMD between KDE of X and U.

    Parameters
    ----------
    X: ndarray --
    Y: ndarrya --
    sigma: float

    Returns
    -------
    mmd: float
    """
    n, d = Y.shape
    m, d2 = X.shape
    assert d == d2

    xy = np.vstack((X, Y))

    k = rbf_kernel(xy, xy, sigma=sigma) + np.eye(n+m)*1e-5
    k_x = k[:m, :m]
    k_y = k[m:, m:]
    k_xy = k[:m, m:]
    # The diagonals are always 1 (up to numerical error, this is (3) in Gretton et al.)
    # note that their code uses the biased (and differently scaled mmd)
    if m == 1:
        mmd = (k_y - np.eye(n)).sum() / (n * (n - 1)) - 2 * k_xy.sum() / (n * m)
    elif n == 1:
        mmd = (k_x - np.eye(m)).sum() / (m * (m - 1)) - 2 * k_xy.sum() / (n * m)
    else:
        mmd = (k_x - np.eye(m)).sum() / (m * (m - 1)) + (k_y - np.eye(n)).sum() / (n * (n - 1)) - 2 * k_xy.sum() / (n * m)
    return np.sqrt(mmd)

def compute_mmd_squared(X, Y, sigma):
    """
    Compute MMD between KDE of X and U.

    Parameters
    ----------
    X: ndarray --
    Y: ndarrya --
    sigma: float

    Returns
    -------
    mmd: float
    """
    n, d = Y.shape
    m, d2 = X.shape
    assert d == d2

    xy = np.vstack((X, Y))

    k = rbf_kernel(xy, xy, sigma=sigma)     # + np.eye(n+m)*1e-5
    k_x = k[:m, :m]
    k_y = k[m:, m:]
    k_xy = k[:m, m:]

    mmd = k_x.sum() / (m**2) + k_y.sum() / (n**2) - 2 * k_xy.sum() / (n * m)
    return mmd


def compute_bootstrap_rkhs_radius(xi, confidence_level=0.95, bootstrap_samples=1000,
                                  kernel_param=None, plot=False, fig_dir=None):
    # print(f'Computing rkhs radius from {int(confidence_level*100)}% bootstrap CI ...')
    """

    Note that we should always use a characteristic kernel, i.e., RBF kernel
    Parameters
    ----------
    xi: ndarray
        Samples of Random Variable
    confidence_level: float
    bootstrap_samples: int
        Number of bootstrap samples
    plot: bool
    fig_dir: str

    Returns
    -------
    rkhs_radius: float
        Bootstrap estimation of radius
    """
    if kernel_param is None:
        kernel_param = {'kernel': 'rbf'}

    n = np.shape(xi)[0]
    m = n
    gram_matrix = compute_gram_matrix(xi, param=kernel_param)
    samples = np.random.choice(np.arange(0, m), size=[bootstrap_samples, m], replace=True)

    mmd = []
    for indices in samples:
        k_x = gram_matrix
        k_xy = gram_matrix[indices, :]
        k_y = k_xy[:, indices]
        mmd_val = k_x.sum() / (n ** 2) + k_y.sum() / (m ** 2) - 2 * k_xy.sum() / (n * m)
        mmd.append(np.sqrt(mmd_val))

    mmd = np.sort(mmd)
    rkhs_radius = mmd[int(np.ceil(len(mmd) * confidence_level))]
    # rate_value = np.sqrt(1/n) + np.sqrt(2 * np.log(1 / 0.05) / n)
    # print(f'Setting RKHS radius to {int(confidence_level*100)}% bootstrap value {rkhs_radius}. '
    #   f'Value based on MMD-rate would have been {rate_value}.')
    if plot:
        plt.rcParams.update(NEURIPS_RCPARAMS)
        plt.style.use('seaborn')
        fig, ax = plt.subplots(1, 1, figsize=(LINE_WIDTH, LINE_WIDTH/2))
        ax.hist(mmd, bins=20, color='b', edgecolor='k', alpha=0.5)
        ax.axvline(rkhs_radius, color='r', linestyle='dashed', linewidth=2)
        min_ylim, max_ylim = plt.ylim()
        text = r'{perc:.0%} CI: {eps:.2f}'.format(perc=confidence_level, eps=rkhs_radius)
        ax.text(rkhs_radius * 1.1, max_ylim * 0.9,
                text.replace('%', r'\%'),
                fontsize='large')
        ax.set_xlabel('MMD estimate')
        ax.set_title("{} Samples".format(n))
        plt.tight_layout()
        plt.savefig(fig_dir / 'MMD_hist_{}.pdf'.format(n), dpi=300)
        plt.show()
    return rkhs_radius


def compute_gram_matrix(x, param):
    """Computes kernel gram matrix of data x using median heuristic and rbf kernel"""
    if param['kernel'] == 'rbf':
        if len(x.shape) < 2:
            x = np.expand_dims(x, axis=1)
        kernel_width, _ = median_heuristic(x, x)
        kernel_matrix = rbf_kernel(x, x, kernel_width)
    elif param['kernel'] == 'linear':
        kernel_matrix = linear_kernel(x, x, param['kernel_param'])
    elif param['kernel'] == 'poly':
        # use a polynomial kernel
        kernel_matrix = polynomial_kernel(x, x, param['degree'])
    else:
        raise ValueError
    return kernel_matrix


def mmd_eps(n_sample, alpha=0.05, ckern=1.0):
    '''compute the mmd guarantee
    alpha = 0.05 # significance level
    ckern = 1.0 # constant due to kernel
    '''
    eps_opt = (1 + np.sqrt(2 * np.log(1 / alpha))) * np.sqrt(ckern / n_sample)
    return eps_opt


def median_heuristic(X, Y):
    '''
    the famous kernel median heuristic
    :param X:
    :param Y:
    :return:
    '''
    distsqr = euclidean_distances(X, Y, squared=True)
    kernel_width = np.sqrt(0.5 * np.median(distsqr))

    '''in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)'''
    kernel_gamma = 1.0 / (2 * kernel_width ** 2)

    return kernel_width, kernel_gamma


def compute_scenarionumber(beta, n_control_vars, p_chance):
    """
    Compute the min. number of scenarios for a given confidence level.

    For a given confidence level beta, probability for individual chance
    constraints and the number of control variables,
    compute the minimum number of scenarios according to
    Hewing, Zeilinger ICSL 2020.
    """
    N = 2 * ((n_control_vars - 1) * np.log(2) - np.log(beta)) / (1 - p_chance)
    return N


def rkhs_func(w, X, x, kernel_param):
    """
    Evaluate rkhs function of form f(x) = alpha.T * K(X, x).

    w: array-like -- weights of rkhs function (n_samples, 1)
    X: ndarray -- support points of rkhs function (n_samples, n_features)
    x: ndarray -- evaluation location for rkhs function (n_points, n_features)
    """
    if kernel_param['kernel'] == 'rbf':
        if "kernel_param" not in kernel_param:
            sigma, _ = median_heuristic(X, X)
        else:
            sigma = kernel_param['kernel_param']
        K = rbf_kernel(X, x, sigma)
    elif kernel_param['kernel'] == 'linear':
        K = linear_kernel(X, x, kernel_param['kernel_param'])
    elif kernel_param['kernel'] == 'poly':
        # use a polynomial kernel
        K = polynomial_kernel(X, X, kernel_param['degree'])
    else:
        raise ValueError

    if w.shape[0] > X.shape[0]:
        K = np.vstack((K, np.ones(K.shape[0])))
    f = K.T @ w
    return f


def rkhs_func_exp(w, X, x, kernel_param):
    """
    Evaluate rkhs function of form f(x) = alpha.T * K(X, x).

    w: array-like -- weights of rkhs function (n_samples, 1)
    X: ndarray -- support points of rkhs function (n_samples, n_features)
    x: ndarray -- support points of rkhs function (n_eval, n_features)
    """
    if kernel_param['kernel'] == 'rbf':
        K = rbf_kernel(X, x, kernel_param['kernel_param'])
    elif kernel_param['kernel'] == 'linear':
        K = linear_kernel(X, x, kernel_param['kernel_param'])
    elif kernel_param['kernel'] == 'poly':
        # use a polynomial kernel
        K = polynomial_kernel(X, X, kernel_param['degree'])
    else:
        raise ValueError

    if w.shape[0] > X.shape[0]:
        K = np.vstack((K, np.ones(K.shape[0])))
    K = K @ np.ones(x.shape[0]) / x.shape[0]
    f = K @ w
    return f


def rkhs_norm_squared(w, X, **param):
    """
    w: array-like -- weights of rkhs function (n_samples, 1)
    X: ndarray -- samples
    param: dict
        sigma: float -- bandwidth of kernel
    """
    if param['kernel'] == 'rbf':
        K = rbf_kernel(X, X, param['kernel_param'])
    elif param['kernel'] == 'linear':
        K = linear_kernel(X, X, param['kernel_param'])
    elif param['kernel'] == 'poly':
        # use a polynomial kernel
        K = polynomial_kernel(X, X, param['degree'])
    else:
        raise ValueError

    if w.shape[0] > X.shape[0]:
        quad = w[:-1] @ K @ w[:-1]
    elif w.shape[0] == X.shape[0]:
        quad = w @ K @ w
    return quad


def rkhs_norm(w, X, **param):
    """
    w: array-like -- weights of rkhs function (n_samples, 1)
    X: ndarray -- samples
    param: dict
        sigma: float -- bandwidth of kernel
    """
    if param['kernel'] == 'rbf':
        K = rbf_kernel(X, X, param['kernel_param'])
    elif param['kernel'] == 'linear':
        K = linear_kernel(X, X, param['kernel_param'])
    elif param['kernel'] == 'poly':
        # use a polynomial kernel
        K = polynomial_kernel(X, X, param['degree'])
    else:
        raise ValueError
    L = cholesky_decomposition(K)
    norm = cp.norm(L @ w)
    return norm


def cvx_rbf_kernel(x, y, sigma):
    """Wrapper for the RBF kernel from sklearn."""
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    gamma = 1 / (2 * sigma ** 2)
    if y.shape[1] == 1:
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(cp.exp(-cp.sum_squares(x[i, :] - y[:, 0])))
        K = cp.vstack(tmp)
    else:
        K = cp.exp(cp.sum_squares(x - y) * gamma)
    return K


def rbf_kernel(x, y, sigma):
    """Wrapper for the RBF kernel from sklearn."""
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    gamma = 1 / (2 * sigma ** 2)
    K = sklearn_kernel(x, y, gamma)
    # K = (K + K.T) / 2
    # assert np.all(K == K.T), "Something is wrong here!{}".format(K)
    return K


def polynomial_kernel(x, y, d):
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    K = (1 + (x @ y.T)**d)
    return K


def linear_kernel(x, y, c):
    if len(x.shape) < 2:
        x = np.expand_dims(x, axis=1)
    if len(y.shape) < 2:
        y = np.expand_dims(y, axis=1)
    K = x @ y.T + c
    return K
