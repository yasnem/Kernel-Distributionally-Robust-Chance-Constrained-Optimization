from drccp.cvx_drccp import CVaRRelaxation, RiskConstrainedProgram, SOSCVaRRelaxation
from drccp.drccp_tools import x_constraints, get_linear_objective
from drccp_utils.distributions import GaussianDistribution
from drccp_utils.rkhs_utils import rkhs_func, compute_bootstrap_rkhs_radius
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH

import argparse
import numpy as np
import cvxpy as cp
import sympy as sp
import dill as pickle
from datetime import datetime
from pathlib import Path
np.set_printoptions(suppress=True)


# Define constraint function
def cvx_f_constraint(x, X):
    """
    Constraint function

    x: cp.Variable -- decision variable (dim,)
    X: ndarray -- Samples (n_samples, dim)
    """
    # f = 0.05 * cp.matmul(X, x) - 1 + 0.5 * np.sin(0.75 * np.sum(X, axis=1)).reshape(-1, 1)
    # f = cp.square(X @ x) - 0.5 + 0.35 * np.sin(0.75 * np.sum(X, axis=1)).reshape(-1, 1)
    # f = 0.01 * cp.square(X @ x) - np.power(np.linalg.norm(X, axis=1), 0.7).reshape(-1, 1)
    f = cp.square(X @ x) - 1
    return f


def solve_cvar_problem(obj, x_const, cc, dim_x, kernel, xis, alpha, epsilon, test_xi,
                       rkhs_method, **kwargs):
    if epsilon == 0:
        cvar_prog = RiskConstrainedProgram(objective=obj,
                                           x_constraints=x_const,
                                           chance_constraints=cc,
                                           dim_x=dim_x)
        obj_val, x_val, emp_cvar = cvar_prog.solve_problem(samples=xis,
                                                           alpha=alpha,
                                                           test_xi=test_xi)
    else:
        # cvar_prog = CVaRRelaxation(objective=obj,
        #                            x_constraints=x_const,
        #                            chance_constraints=cc,
        #                            dim_x=dim_x,
        #                            kernel_param={'kernel': kernel},
        #                            rkhs_method=rkhs_method)
        # obj_val, x_val, emp_cvar, weights, g0 = cvar_prog.solve_problem(samples=xis,
        #                                                                 alpha=alpha,
        #                                                                 epsilon=epsilon,
        #                                                                 test_xi=test_xi,
        #                                                                 **kwargs)
        kernel_param = {
            'kernel': 'poly',
            'degree': kwargs['degree']
        }
        sos_prog = SOSCVaRRelaxation(objective=obj,
                                     x_constraints=x_const,
                                     chance_constraints=cc,
                                     dim_x=dim_x,
                                     xi_sym=kwargs['xi_sym'],
                                     x_sym=kwargs['x_sym'],
                                     kernel_param=kernel_param)

        obj_val, x_val, emp_cvar, weights, g0 = sos_prog.solve_problem(samples=xis,
                                                                       alpha=alpha,
                                                                       epsilon=epsilon,
                                                                       test_xi=test_xi)
        # create_figure(x_val, weights, g0, comb_pts, cc, {'kernel': kernel}, epsilon)
    if obj_val is None:
        print("Optimization failed.")
        return obj_val, x_val, 1, epsilon, xis.shape[0]
    p_constraint = np.sum(np.asarray(sos_prog.eval_cc(x_val, test_xi)) <= 0) / np.shape(test_xi)[0]
    return obj_val, x_val, p_constraint, emp_cvar, epsilon, xis.shape[0]


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--risk_level', type=float, default=0.1)
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--visualize', dest='compute', action='store_false')
parser.add_argument('--exp_name', type=str, default='cvar_data')
parser.set_defaults(compute=True)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.compute:
        current_time = datetime.now()
        time_stamp = current_time.timestamp()
        date_time = datetime.fromtimestamp(time_stamp)
        time_str = date_time.strftime("%d_%m_%H_%M")
        exp_name = args.exp_name + '_' + time_str + '_' + '.pk'
    else:
        exp_name = args.exp_name + '.pk'
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / exp_name

    # Define objective
    # For some reason I need to import the objective and constraint function from a different file.
    # Seems to be an issue with multiprocessing and python 3.9.
    # Therefore use multiprocess and dill.
    if args.compute:
        # dist = UniformDistribution(args.dim, lb=-2, ub=2)
        mean = np.zeros(args.dim)
        cov = np.zeros((args.dim, args.dim))
        np.fill_diagonal(cov, np.arange(0.5, 0.5*(args.dim + 1), 0.5))
        dist = GaussianDistribution(args.dim, mean=mean, std=cov)
        alpha = args.risk_level
        eps_lst = [0.1, 'bootstrap']
        n_samples = 40
        test_xi = dist.sample(10000)

        objective = get_linear_objective(args.dim)
        np.random.seed(10)
        X = dist.sample(n_samples)

        xi_sym = sp.symbols('xi1:%d' % (args.dim + 1))
        xi_arr = np.array([xi for xi in xi_sym])
        x_sym = sp.symbols('x1:%d' % (args.dim + 1))
        x_arr = np.array([x for x in x_sym])
        f_const = (-xi_arr**4 + 4*xi_arr**3 + 6*xi_arr**2) @ x_arr - 1

        sol = solve_cvar_problem(objective, x_constraints, f_const,
                                 args.dim, args.kernel, X, alpha, eps_lst[1],
                                 test_xi, 'rand_feature', n_rand_feat=1000,
                                 xi_sym=xi_sym, x_sym=x_sym, degree=4)
        print('objective: ', sol[0])
        print('x_value: ', sol[1])
        print('chance constraint: ', sol[2])
        print('empirical CVaR: ', sol[3])
