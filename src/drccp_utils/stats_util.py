import numpy as np
from scipy.spatial import ConvexHull

from drccp_utils.rkhs_utils import compute_mmd, rbf_kernel, rkhs_dist
from drccp_utils.distributions import UniformDistribution


def sample_zetas_convex_hull(xi, n_sample):
    dim_xi = np.shape(xi)[1]

    hull = ConvexHull(xi)
    bbox = [hull.min_bound, hull.max_bound]

    zetas = np.empty([n_sample, dim_xi])
    for i in range(dim_xi):
        zetas[:, i] = np.random.uniform(low=bbox[0][i], high=bbox[1][i], size=n_sample)
    return zetas


def grid_around_data_support(n_points, X_emp, delta=0):
    lb = np.min(X_emp, axis=0) - delta
    ub = np.max(X_emp, axis=0) + delta
    _, dim = X_emp.shape
    if n_points == 0:
        return np.empty((0, dim))
    eval_points = np.linspace(lb, ub, num=n_points)
    grid = np.meshgrid(*eval_points.T)
    for i in range(len(grid)):
        grid[i] = grid[i].flatten()
    eval_points = np.vstack(grid).T
    return eval_points


def sample_around_data_support(n_points, X_emp, delta=0):
    lb = np.min(X_emp, axis=0)
    ub = np.max(X_emp, axis=0)
    _, dim = X_emp.shape
    if n_points == 0:
        return np.empty((0, dim))
    dist = UniformDistribution(dim,
                               lb=lb - delta * np.ones_like(lb),
                               ub=ub + delta * np.ones_like(ub))
    return dist.sample(n_points)
