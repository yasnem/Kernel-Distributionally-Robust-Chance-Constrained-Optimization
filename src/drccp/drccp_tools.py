import numpy as np
import cvxpy as cp
from sklearn.model_selection import KFold
from drccp.cvx_drccp import CVaRRelaxation


def get_linear_objective(dim):
    c = np.arange(1, 1 + dim * 0.5, 0.5)

    def objective(x):
        return c@x

    return objective


def x_constraints(x, dim=None):
    return [
        x >= 0,
        cp.sum(x) <= 1
    ]


def solve_cvar_prob(obj, x_const, cc, dim_x, kernel, Xi, alpha, eps, n_supp, Xi_test):
    cvar_prog = CVaRRelaxation(objective=obj,
                               x_constraints=x_const,
                               chance_constraints=cc,
                               dim_x=dim_x,
                               kernel_param={'kernel': kernel})
    obj_val, x_val = cvar_prog.solve_problem(samples=Xi,
                                             alpha=alpha,
                                             epsilon=eps,
                                             n_supp_points=n_supp)
    p_constraint = np.sum(np.asarray(cvar_prog.chance_constraint(x_val, Xi_test).value) <= 0) / np.shape(Xi_test)[0]
    return eps, obj_val, 1 - p_constraint


def rkhs_radius_CV(pool, obj, x_const, cc, dim_x, kernel, xis, alpha):
    kf = KFold(n_splits=4)
    epsilons = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
    cv_vio = {eps: [] for eps in epsilons}
    cv_obj = {eps: [] for eps in epsilons}
    results = []
    for eps in epsilons:
        for train_idx, test_idx in kf.split(xis):
            X_train, X_test = xis[train_idx], xis[test_idx]
            results.append(pool.apply_async(solve_cvar_prob,
                                            (obj, x_const, cc, dim_x, kernel,
                                             X_train, alpha, eps, xis.shape[0],
                                             X_test)))
    for res in results:
        eps, obj_val, p_constraint = res.get()
        if obj_val is None:
            continue
        cv_obj[eps].append(obj_val)
        cv_vio[eps].append(p_constraint)

    return cv_obj, cv_vio