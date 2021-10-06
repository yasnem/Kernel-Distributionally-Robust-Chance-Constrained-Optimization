from drccp.cvx_drccp import CVaRRelaxation, RiskConstrainedProgram
from drccp.drccp_tools import (rkhs_radius_CV, x_constraints,
                               get_linear_objective)
from drccp_utils.distributions import GaussianDistribution
from drccp_utils.rkhs_utils import rkhs_func
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH

import argparse
import numpy as np
import cvxpy as cp
from multiprocess import Pool, cpu_count
import dill as pickle
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


def create_figure(x_sol, w_sol, g0, comb_points, f_constraint, kernel_param, eps):
    """Create Majorization figure of optimized RKHS-function."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(NEURIPS_RCPARAMS)
    plt.style.use('seaborn')
    lb = np.min(comb_points, axis=0) - 2
    ub = np.max(comb_points, axis=0) + 2
    mean = np.mean(comb_points, axis=0)
    dim = len(x_sol)
    # x_sol *= 0.8
    print(g0, w_sol)
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
        print(g_rkhs, f_val, pts)
        plt.plot(pts, f_val, label=r'$f(x, \xi)$')
        plt.plot(pts, g_rkhs, 'k', label=r'$g(\xi)$')
        plt.plot(pts, np.array(f_val > 0), 'r--', label=r'$1(f(x, \xi))$')
        plt.axvline(lb[dim-i-1]+2, color='k', linestyle='dashed', linewidth=2)
        plt.axvline(ub[dim-i-1]-2, color='k', linestyle='dashed', linewidth=2)
        plt.xlabel(r'$\xi$')
        plt.title("Epsilon: {}".format(eps))
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()


def plot(data, name, data_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    fig, ax = plt.subplots(1, 2, figsize=(LINE_WIDTH, LINE_WIDTH/2))

    for j, (key, quantity) in enumerate(data.items()):
        for i, (rkhs_radius, results) in enumerate(quantity.items()):
            n_samples = []
            res_val = []
            for n, res in results.items():
                if np.alltrue(np.isfinite(np.asarray(res).astype(np.double))):
                    res_val.append(res)
                    n_samples.append(n)
            if True:
                mean = [np.mean(rollouts) for rollouts in res_val]
                std = [np.std(rollouts)/np.sqrt(len(rollouts)) for rollouts in
                       res_val]
                if key == 'p_constraint':
                    mean = 1 - np.asarray(mean)
                if rkhs_radius == 'bootstrap':
                    label = r'$\mathcal{P}$ bootstrap'
                elif rkhs_radius == 'rate':
                    label = r'$\mathcal{P}$ rate'
                elif rkhs_radius == 0:
                    label = r'$\mathcal{P} = \{\hat{P}_n\}$'
                elif rkhs_radius == 'cv':
                    label = 'CV'
                else:
                    label = rf'$\varepsilon$ = {np.round(rkhs_radius, 5)}'
                ax[j].plot(n_samples, mean, label=label,
                           color=colors[i], marker=marker[i], ms=10)
                ax[j].fill_between(n_samples,
                                   np.subtract(mean, std),
                                   np.add(mean, std),
                                   alpha=0.2,
                                   color=colors[i])
                ax[j].set_xscale('log')

        ax[j].set_xlabel('sample size')
        if key == 'p_constraint':
            ax[j].set_ylabel(r'$P(f(x,\xi) > 0)$')
            ax[j].set_title(r'Constraint violation, $\alpha=0.05$')
        if key == 'x_returns':
            # ax[j].set_ylabel(r'$\max c^T x$')
            ax[j].set_ylabel(r'$c^Tx$')
            ax[j].set_title('Objective')
        elif key == 'cvar':
            ax[j].set_ylabel(r'$P_0-$CVaR')
            ax[j].set_title('Constraint')
        else:
            raise ValueError("The following key does not exist: ", key)

    plt.legend()
    plt.tight_layout()
    fig_name = "portfolio_{}.pdf".format(name)
    fig_path = data_dir / fig_name
    plt.savefig(fig_path, dpi=300)
    plt.show()


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
        cvar_prog = CVaRRelaxation(objective=obj,
                                   x_constraints=x_const,
                                   chance_constraints=cc,
                                   dim_x=dim_x,
                                   kernel_param={'kernel': kernel},
                                   rkhs_method=rkhs_method)
        obj_val, x_val, emp_cvar, weights, g0 = cvar_prog.solve_problem(samples=xis,
                                                                        alpha=alpha,
                                                                        epsilon=epsilon,
                                                                        test_xi=test_xi,
                                                                        **kwargs)
        # create_figure(x_val, weights, g0, comb_pts, cc, {'kernel': kernel}, epsilon)
    if obj_val is None:
        print("Optimization failed.")
        return obj_val, x_val, 1, 1, epsilon, xis.shape[0]
    p_constraint = np.sum(np.asarray(cvar_prog.chance_constraint(x_val, test_xi).value) <= 0) / np.shape(test_xi)[0]
    return obj_val, x_val, p_constraint, emp_cvar, epsilon, xis.shape[0]


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=3)
parser.add_argument('--risk_level', type=float, default=0.1)
parser.add_argument('--n_runs', type=int, default=16)
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--rkhs_method', type=str, default='rand_feature')
parser.add_argument('--n_rand_feat', type=int, default=1000)
parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--visualize', dest='compute', action='store_false')
parser.add_argument('--exp_name', type=str, default='cvar_rff')
parser.set_defaults(compute=True)


if __name__ == "__main__":
    args = parser.parse_args()
    exp_data = args.exp_name + '_data.pk'
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    file_path = data_dir / exp_data

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
        eps_lst = [0, 'bootstrap', 'rate']
        n_samples = [40, 100, 200, 400, 800, 1200]
        test_xi = dist.sample(100000)
        violations_cvar = {eps: {n_sample: [] for n_sample in n_samples} for eps in eps_lst}
        objectives_cvar = {eps: {n_sample: [] for n_sample in n_samples} for eps in eps_lst}
        emp_cvar_dict = {eps: {n_sample: [] for n_sample in n_samples} for eps in eps_lst}

        objective = get_linear_objective(args.dim)
        results = []
        pool = Pool(cpu_count())
        for n_sample in n_samples:
            for i in range(args.n_runs):
                np.random.seed(123 + i)
                X = dist.sample(n_sample)
                for j, eps in enumerate(eps_lst):
                    if eps == 'cv':
                        cv_obj, cv_vio = rkhs_radius_CV(pool=pool,
                                                        obj=objective,
                                                        x_const=x_constraints,
                                                        cc=cvx_f_constraint,
                                                        dim_x=args.dim,
                                                        kernel=args.kernel,
                                                        xis=X,
                                                        alpha=alpha)
                        max_eps = 0
                        max_obj = 0
                        for eps in cv_vio:
                            mean_vio = np.mean(cv_vio[eps])
                            mean_obj = np.mean(cv_obj[eps])
                            if mean_vio <= alpha and mean_obj > max_obj:
                                max_eps = eps
                                max_obj = mean_obj
                        obj_cv, x_cv, vio_cv, emp_cvar, _ = solve_cvar_problem(obj=objective,
                                                                               x_const=x_constraints,
                                                                               cc=cvx_f_constraint,
                                                                               dim_x=args.dim,
                                                                               kernel=args.kernel,
                                                                               xis=X,
                                                                               alpha=alpha,
                                                                               epsilon=max_eps,
                                                                               test_xi=test_xi,
                                                                               rkhs_method=args.rkhs_method,
                                                                               n_rand_feat=args.n_rand_feat)
                        objectives_cvar['cv'][n_sample].append(obj_cv)
                        violations_cvar['cv'][n_sample].append(vio_cv)
                        emp_cvar_dict['cv'][n_sample].append(emp_cvar)
                        continue
                    results.append(pool.apply_async(solve_cvar_problem,
                                                    (objective, x_constraints,
                                                     cvx_f_constraint, args.dim,
                                                     args.kernel, X, alpha,
                                                     eps, test_xi, args.rkhs_method,
                                                     ), kwds={'n_rand_feat':
                                                              args.n_rand_feat}))

        for res in results:
            obj, x, vio, emp_cvar, eps, n_sample = res.get()
            if obj is None:
                continue
            objectives_cvar[eps][n_sample].append(obj)
            violations_cvar[eps][n_sample].append(vio)
            emp_cvar_dict[eps][n_sample].append(emp_cvar)
        data_cvar = {
            'cvar': emp_cvar_dict,
            # 'p_constraint': violations_cvar,
            'x_returns': objectives_cvar
        }
        with file_path.open('wb') as fid:
            pickle.dump(data_cvar, fid)
    else:
        with file_path.open('rb') as fid:
            data_cvar = pickle.load(fid)
    plot(data_cvar, args.exp_name, data_dir)
