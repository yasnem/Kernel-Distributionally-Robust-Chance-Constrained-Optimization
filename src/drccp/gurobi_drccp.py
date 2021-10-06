import numpy as np
import gurobipy as gp
from gurobipy import GRB
from drccp_utils.rkhs_utils import rkhs_func, rkhs_norm_squared, rkhs_func_exp


def gurobi_exact(f_constraint, X, c, alpha, epsilon, M, sigma, supp_points):
    n_samples, dim = X.shape
    supp_points = np.vstack((supp_points, X))
    # supp_points = X
    n_supp = supp_points.shape[0]
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as model:
            x = model.addMVar(dim, name='x')
            w = model.addMVar(n_supp, name='weights')
            mu = model.addMVar((2, n_supp), vtype=GRB.BINARY, name='switch')
            t = model.addMVar(1, name='slack')

            # Set objective
            model.setObjective(c @ x, GRB.MINIMIZE)
            model.params.NonConvex = 2

            # Constraint variables
            f_const = f_constraint(x, supp_points)
            g_rkhs = rkhs_func(w, supp_points, supp_points, sigma=sigma)
            Eg_rkhs = rkhs_func_exp(w, supp_points, X, sigma=sigma)
            g_norm = rkhs_norm_squared(w, supp_points, sigma=sigma)

            model.addConstr(x >= 0)
            model.addConstr(f_const <= (1 - mu[1, :]) * M)
            model.addConstr(f_const >= -(1 - mu[0, :]) * M)
            model.addConstr(g_rkhs >= mu[0, :])
            model.addConstr(Eg_rkhs + epsilon * t <= alpha)
            model.addConstr(g_norm <= t @ t)
            model.addConstr(t >= 0)

            ones = np.ones(2)
            for i in range(supp_points.shape[0]):
                model.addConstr(mu[:, i] @ ones == 1)

            model.optimize()

            # for v in model.getVars():
            #     print('%s %g' % (v.varName, v.x))
            decision_variable = (np.eye(x.shape[0]) @ x).getValue()
            print("Decision var: ", decision_variable)
            # print("Constraint: ", f_constraint(x, X).getValue())
            print("RKHS function: ", g_rkhs.getValue())
            print("RKHS function exp: ", Eg_rkhs.getValue())
            print("RKHS norm: ", np.sqrt(g_norm.getValue()))
            print('Obj: %g' % model.objVal)
            return decision_variable


def gurobi_scen(f_constraint, X, c):
    n_samples, dim = X.shape
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as scenario_model:
            x = scenario_model.addMVar(dim, name='x')

            # Set objective
            scenario_model.setObjective(c @ x, GRB.MINIMIZE)
            scenario_model.params.NonConvex = 2

            scenario_model.addConstr(x >= 0)
            scenario_model.addConstr(f_constraint(x, X) <= 0)

            scenario_model.optimize()

            print("Decision var: ", (np.eye(x.shape[0]) @ x).getValue())
            print('Obj: %g' % scenario_model.objVal)
