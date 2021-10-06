import cvxpy as cp
import numpy as np
import argparse

from drccp_utils.distributions import PolytopicGaussianDistribution, PolytopicUniformDistribution
from drccp_utils.rkhs_utils import linear_kernel, rkhs_func, rkhs_norm, rkhs_func_exp
from drccp_utils.stats_util import grid_around_data_support

def constraint(x, X):
    # x2 = cp.square(x)
    return X @ x - 1.5

def constraint_x(x):
    # return cp.square(x)
    return x


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=5)
parser.add_argument('--n_samples', type=int, default=10)
parser.add_argument('--n_supp', type=int, default=0)
parser.add_argument('--risk_level', type=float, default=0.1)
parser.add_argument('--kernel', type=str, default='linear')
parser.add_argument('--kernel_param', type=float, default=1)

parser.add_argument('--fixed_seed', dest='fix_seed', action='store_true')
parser.add_argument('--not_fixed_seed', dest='fix_seed', action='store_false')
parser.add_argument('--random', dest='rand_obj', action='store_true')
parser.add_argument('--not_random', dest='rand_obj', action='store_false')
parser.set_defaults(fix_seed=False, rand_obj=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dist = PolytopicGaussianDistribution(args.dim,
                                         0.2 * np.ones(args.dim),
                                         std=1.2 * np.eye(args.dim),
                                         A=-np.eye(args.dim),
                                         b=np.zeros(args.dim))
    samples = dist.sample(args.n_samples)
    zetas = grid_around_data_support(args.n_supp, samples, 0.2)
    psi = np.vstack((samples, zetas))

    epsilon = 0.1

    if args.rand_obj:
        c = np.random.normal(size=(1, args.dim))
    else:
        c = np.linspace(-5, 5, args.dim)

    def objective(x):
        return c @ x

    x = cp.Variable(shape=(args.dim, 1), pos=True, name='decision')
    t = cp.Variable(nonneg=True, name='CVaR')
    beta = cp.Variable(shape=(args.n_samples + args.n_supp**args.dim, 1), name='weights')
    y1 = cp.Variable(shape=(len(dist.b), 1), nonneg=True, name='lin_dual1')
    y2 = cp.Variable(shape=(len(dist.b), 1), nonneg=True, name='lin_dual2')
    xi_primal = cp.Variable(shape=(args.dim, 1), name='lin_primal')
    g0 = cp.Variable(1, name='g0')

    obj = objective(x)
    f_const = constraint(x, psi)
    g_rkhs = rkhs_func(w=beta,
                       X=psi,
                       x=samples,
                       kernel=args.kernel,
                       kernel_param=args.kernel_param)
    Eg_rkhs = rkhs_func_exp(beta, psi, samples,
                            kernel=args.kernel,
                            kernel_param=args.kernel_param)
    g_norm = rkhs_norm(w=beta,
                       X=psi,
                       kernel=args.kernel,
                       kernel_param=args.kernel_param)

    constraints_dual = [
        g0 + Eg_rkhs + epsilon * g_norm <= t * args.risk_level,
        dist.b.T @ y1 + t - g0 + cp.sum(beta) <= 0,
        dist.A.T @ y1 + beta.T @ psi - constraint_x(x) == 0,
        dist.b.T @ y2 - g0 + cp.sum(beta) <= 0,
        dist.A.T @ y2 + beta.T @ psi == 0,
        cp.sum(x) == 1
    ]

    # constraints_primal = [
    #     g0 + Eg_rkhs + epsilon * g_norm <= t * args.risk_level,
    #     constraint_x(x).T @ xi_primal - beta.T @ psi @ xi_primal + t - g0 <= 0,
    #     beta.T @ psi @ xi_primal + g0 >= 0,
    #     dist.A @ xi_primal <= dist.b.reshape((args.dim, 1)),
    #     cp.sum(x) == 1
    # ]

    K = linear_kernel(samples, samples, -3)
    print('test')
    problem = cp.Problem(cp.Minimize(obj), constraints_dual)
    problem.solve(solver=cp.MOSEK, verbose=True)
    print(x.value)
    print('Done')
