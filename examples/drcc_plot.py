import numpy as np
from drccp.cvx_drccp import cvx_exact_plot
from drccp_utils.distributions import GaussianDistribution
from drccp_utils.rkhs_utils import rkhs_func, compute_bootstrap_rkhs_radius
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH

import argparse
import cvxpy as cp
from pathlib import Path

np.set_printoptions(suppress=True)


# Define constraint function
def cvx_f_constraint(x, X):
    """
    Constraint function

    x: cp.Variable -- decision variable (dim,)
    X: ndarray -- Samples (n_samples, dim)
    """
    #     f = cp.exp(0.1 * (x[0, :] + X[:,0])) + x[1, :] + X[:, 1] -10

    # f = 0.05 * cp.matmul(X, x) - 1 + 0.5 * np.sin(0.75 * np.sum(X, axis=1)).reshape(-1, 1)
    # f = cp.square(X @ x) - 0.5 + 0.35 * np.sin(0.75 * np.sum(X, axis=1)).reshape(-1, 1)
    f = cp.exp(-2*X**2) - 0.25
    # f = 0.01 * cp.square(X @ x) - np.power(np.linalg.norm(X, axis=1), 0.7).reshape(-1, 1)
    return f


def create_figure(x_sol, w_sol, g0, comb_points, f_constraint, kernel_param, fig_dir):
    """Create Majorization figure of optimized RKHS-function."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(NEURIPS_RCPARAMS)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1, figsize=(8*LINE_WIDTH/12, LINE_WIDTH/3))
    lb = np.min(comb_points, axis=0)
    ub = np.max(comb_points, axis=0)
    mean = np.mean(comb_points, axis=0)
    dim = len(x_sol)
    for i in range(dim):
        # set evaluation points
        n_points = 100
        eval_points = np.zeros((n_points, dim))
        eval_points[:, dim-i-1] = np.linspace(lb[dim-i-1], ub[dim-i-1], num=n_points)
        if dim > 1:
            eval_points[:, i] = mean[i]
        pts = eval_points[:, dim-i-1]
        g_rkhs = rkhs_func(w_sol, comb_points, eval_points, kernel_param=kernel_param) + g0
        f_val = f_constraint(x_sol, eval_points).value
        ax.plot(pts, f_val, label=r'$f(x, \xi)$')
        ax.plot(pts, g_rkhs, 'k', label=r'$g(\xi)$')
        ax.plot(pts, np.array(f_val > 0), 'r--', label=r'$1(f(x, \xi) > 0)$')
        # markerline, stemlines, baseline = plt.stem(comb_points, np.array(f_comb > 0),
        #                                            linefmt='r--', markerfmt='ro',
        #                                            label=r'$1(f(x, \xi))$')
        # plt.setp(baseline, linestyle='')
        # plt.axvline(lb[dim-i-1]+2, color='k', linestyle='dashed', linewidth=2)
        # plt.axvline(ub[dim-i-1]-2, color='k', linestyle='dashed', linewidth=2)
        # plt.title("Epsilon: {}".format(np.round(eps, 3)))
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel(r'$\xi$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / 'majorize.pdf', dpi=300)
        plt.show()
        plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--risk_level', type=float, default=0.2)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--conf_lvl', type=float, default=0.95)


if __name__ == "__main__":
    args = parser.parse_args()
    # Define objective
    c = np.linspace(1, 10, args.dim)
    # c = np.random.randint(1, 50, size=args.dim)

    def objective(x):
        return c @ x

    def x_constraints(x):
        return [
            x >= 0,
            cp.sum(x) <= 1
        ]
    mean = np.zeros(args.dim)
    std = np.eye(args.dim)
    dist = GaussianDistribution(args.dim, mean=mean, std=std)
    # dist = UniformDistribution(args.dim, lb=-2, ub=2)
    alpha = args.risk_level
    n_samples = 1000
    samples = dist.sample(n_samples)
    samples = np.linspace(-3, 3, 500).reshape(-1, 1)
    # eps = compute_bootstrap_rkhs_radius(samples, confidence_level=args.conf_lvl)
    eps = 0.001
    x_sol = np.ones((args.dim, 1))/args.dim
    w_sol, g0 = cvx_exact_plot(x_sol, cvx_f_constraint, samples, args.risk_level, eps, args.kernel)
    fig_dir = Path(__file__).parent / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    create_figure(x_sol, w_sol, g0, samples, cvx_f_constraint, {'kernel': args.kernel}, fig_dir)



