import cvxpy as cvx
import numpy as np
from drccp.cvx_drccp import CVaRRelaxation
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import copy


def get_objective_and_constraints(dim_x, random=False):
    if random:
        c = np.random.normal(size=(1, dim_x))
    else:
        c = np.linspace(1, 10, dim_x)
    def objective(x):
        return c @ x

    def constraint_x(x):
        return [cvx.sum(x) <= 1, x >= 0]

    def chance_constraint(x, xi):
        return cvx.square(xi @ x) - 1
    return objective, constraint_x, chance_constraint


def run_rollout(n_sample, xi_std, alpha, epsilon, n_sample_points, seed, test_xi):
    np.random.seed(seed)

    objective, constraint_x, chance_constraint = get_objective_and_constraints(dim_x=dim_x, random=False)
    xi = np.random.normal(size=(n_sample, dim_x), scale=xi_std)

    solver = CVaRRelaxation(objective=objective, x_constraints=constraint_x,
                            chance_constraints=chance_constraint, dim_x=dim_x, kernel_param={'kernel': 'rbf'})
    status, x = solver.solve_problem(samples=xi, alpha=alpha, epsilon=epsilon, n_supp_points=n_sample_points)
    p_constraint = np.sum(np.asarray(solver.chance_constraint(x, test_xi).value) <= 0) / np.shape(test_xi)[0]
    x_returns = solver.objective(x)[0]
    return p_constraint, x_returns


def run_parallel(n_rollouts, n_sample, xi_std, alpha, epsilon, n_sample_points, test_xi):
    test_xi_list = [copy.deepcopy(test_xi) for _ in range(n_rollouts)]
    n_sample_list = [copy.deepcopy(n_sample) for _ in range(n_rollouts)]
    xi_std_list = [copy.deepcopy(xi_std) for _ in range(n_rollouts)]
    alpha_list = [copy.deepcopy(alpha) for _ in range(n_rollouts)]
    epsilon_list = [copy.deepcopy(epsilon) for _ in range(n_rollouts)]
    n_sample_points_list = [copy.deepcopy(n_sample_points) for _ in range(n_rollouts)]
    seeds = np.arange(123, 123+n_rollouts)

    with ProcessPoolExecutor() as ex:
        res = ex.map(run_rollout, n_sample_list, xi_std_list, alpha_list, epsilon_list, n_sample_points_list, seeds, test_xi_list)
    return res


def plot(data):
    plt.rcParams.update(NEURIPS_RCPARAMS)
    sns.set_theme()
    marker = ['v', 'o', 's', 'd', 'p', '*', 'h']
    colors = ['tab:blue', 'tab:red', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:cyan', 'tab:purple']

    fig, ax = plt.subplots(1, 2, figsize=(LINE_WIDTH, LINE_WIDTH/2))

    for j, (key, quantity) in enumerate(data.items()):
        for i, (rkhs_radius, results) in enumerate(quantity.items()):
            n_samples =[]
            res_val = []
            for n, res in results.items():
                if np.alltrue(np.isfinite(np.asarray(res).astype(np.double))):
                    res_val.append(res)
                    n_samples.append(n)
            if True:
                mean = [np.mean(rollouts) for rollouts in res_val]
                std = [np.std(rollouts) for rollouts in res_val]
                if key == 'p_constraint':
                    mean = 1 - np.asarray(mean)
                if rkhs_radius is None:
                    label = 'MMD-bootstrap'
                else:
                    label = rf'$\varepsilon$ = {rkhs_radius}'
                ax[j].plot(n_samples, mean, label=label, color=colors[i], marker=marker[i], ms=10)
                ax[j].fill_between(n_samples,
                                   np.subtract(mean, std),
                                   np.add(mean, std),
                                   alpha=0.2,
                                   color=colors[i])

        ax[j].set_xlabel('sample size')
        if key == 'p_constraint':
            ax[j].set_ylabel(r'$P(f(x,\xi) > 0)$')
            ax[j].set_title(r'Constraint violation, $\alpha=0.05$')
        elif key == 'x_returns':
            ax[j].set_ylabel(r'$\max c^T x$')
            ax[j].set_title('Objective')

    plt.legend()
    plt.tight_layout()
    plt.savefig('investment_experiment.pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    compute = True

    dim_x = 5
    n_samples = [50, 100, 200]
    xi_std = 2.0
    n_runs = 8
    alpha = 0.05
    epsilons = [1e-2, 1e-1, None]
    debug = False

    p_constraint = {epsilon: {n_sample: [] for n_sample in n_samples} for epsilon in epsilons}
    x_returns = {epsilon: {n_sample: [] for n_sample in n_samples} for epsilon in epsilons}
    test_xi = np.random.normal(size=(10000, dim_x), scale=xi_std)

    # objective, constraint_x = get_test_objective_and_constraint(dim_x=dim_x, random=False)
    # constraint = test_constraint_function
    # solver = CVaRKDRO(objective=objective, x_constraints=constraint_x, chance_constraints=constraint, dim_x=dim_x, xi=test_xi)
    # rkhs_radius = solver.rkhs_radius

    if compute:
        for epsilon in epsilons:
            for n_sample in n_samples:
                print(f'Running setup: epsilon={epsilon}, n_sample={n_sample}')
                kwargs = {'test_xi': test_xi, 'n_sample': n_sample, 'xi_std': xi_std,
                          'alpha': alpha, 'epsilon': epsilon, 'n_sample_points': n_sample}

                if debug:
                    results = []
                    for i in range(n_runs):
                        results.append(run_rollout(**kwargs, seed=123 + i))
                    results = run_parallel(n_rollouts=n_runs, **kwargs)
                    results = [res for res in results]
                else:
                    try:
                        results = run_parallel(n_rollouts=n_runs, **kwargs)
                        results = [res for res in results]
                    except:
                        results = [[[None for _ in range(n_runs)], [None for _ in range(n_runs)]]]
                        print('Optimization failed.')

                p_constraint[epsilon][n_sample] = [res[0] for res in results]
                x_returns[epsilon][n_sample] = [res[1] for res in results]

                try:
                    print(rf'P(f(x,\xi) <= 0) = {np.mean(p_constraint[epsilon][n_sample])} \pm {np.std(p_constraint[epsilon][n_sample])}')
                    print(rf'max_x c^T x = {np.mean(x_returns[epsilon][n_sample])} \pm {np.std(x_returns[epsilon][n_sample])}')
                except:
                    pass

        data = {'p_constraint': p_constraint, 'x_returns': x_returns}
        with open('portfolio_data.pk', 'wb') as file:
            pickle.dump(data, file)

    with open('portfolio_data.pk', 'rb') as file:
        data = pickle.load(file)

    plot(data)
