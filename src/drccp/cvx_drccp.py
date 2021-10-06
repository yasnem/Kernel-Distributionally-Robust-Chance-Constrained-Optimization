import cvxpy as cp
import sympy as sp
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from drccp_utils.rkhs_utils import (rkhs_func, rkhs_norm, rkhs_func_exp,
                                    compute_bootstrap_rkhs_radius, median_heuristic,
                                    compute_gram_matrix, cholesky_decomposition, mmd_eps)
from  drccp_utils.sos_utils import create_sos_constraints


class CCP(metaclass=ABCMeta):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x):
        self.objective = objective
        self.x_constraints = x_constraints
        self.chance_constraint = chance_constraints
        self.dim_x = dim_x
        self.constraints = []

    @abstractmethod
    def solve_problem(self, *args, **kwargs):
        pass


class RiskConstrainedProgram(CCP):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x):
        super(RiskConstrainedProgram, self).__init__(objective=objective,
                                                     x_constraints=x_constraints,
                                                     chance_constraints=chance_constraints,
                                                     dim_x=dim_x)

    def solve_problem(self, samples, alpha, test_xi):
        # Define variables
        x = cp.Variable(shape=(self.dim_x, 1), name='x')
        t = cp.Variable(1, name='CVaR')

        # Define objective
        self.constraints = self.x_constraints(x)

        # Define constraints
        n_samples, dim_xi = samples.shape
        f_const = self.chance_constraint(x, samples)
        self.constraints.append(cp.sum(cp.maximum(f_const + t, 0))/n_samples <= t*alpha)

        prob = cp.Problem(objective=cp.Maximize(self.objective(x)),
                          constraints=self.constraints)
        prob.solve(solver="MOSEK")
        if prob.status in ['infeasible', 'unbounded']:
            raise ValueError
        obj_sol = prob.value
        x_sol = x.value

        # compute CVaR
        t = cp.Variable(1, 'CVaR')
        f_cvar = self.chance_constraint(x, test_xi).value
        cvar_obj = cp.sum(cp.maximum(f_cvar + t, 0)) / test_xi.shape[0] - t * alpha
        prob = cp.Problem(objective=cp.Minimize(cvar_obj))
        try:
            prob.solve(solver="MOSEK")
            emp_cvar = prob.value
        except:
            print("Could not compute CVaR")
        return obj_sol, x_sol, emp_cvar


class ScenarioProgram(CCP):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x):
        super(ScenarioProgram, self).__init__(objective=objective,
                                              x_constraints=x_constraints,
                                              chance_constraints=chance_constraints,
                                              dim_x=dim_x)

    def solve_problem(self, samples):
        # Define variables
        x = cp.Variable(self.dim_x, name='x')

        # Define objective
        self.constraints = self.x_constraints(x)

        # Define constraints
        f_const = self.chance_constraint(x, samples)
        self.constraints.append(f_const <= 0)

        prob = cp.Problem(objective=cp.Maximize(self.objective(x)),
                          constraints=self.constraints)
        prob.solve(solver="MOSEK")
        if prob.status == 'infeasible':
            raise ValueError

        return prob.value, x.value


class MIPRelaxation(CCP):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x, kernel, kernel_param):
        super(MIPRelaxation, self).__init__(objective=objective,
                                            x_constraints=x_constraints,
                                            chance_constraints=chance_constraints,
                                            dim_x=dim_x)
        self.kernel = kernel
        self.kernel_param = kernel_param

    def solve_problem(self, samples, alpha, epsilon, M, supp_points):
        """

        Parameters
        ----------
        samples
        alpha
        epsilon
        M
        supp_points

        Returns
        -------

        """
        n_samples, dim_xi = samples.shape
        comb_points = np.vstack((supp_points, samples))
        n_supp = supp_points.shape[0]

        # Define variables
        x = cp.Variable(self.dim_x, name='x')
        w = cp.Variable(n_supp + n_samples, name='weights')
        mu = cp.Variable((2, n_supp + n_samples), boolean=True)
        g0 = cp.Variable(1, name='g0')

        # Define constraints
        f_const = self.chance_constraint(x, comb_points)
        g_rkhs = rkhs_func(w, comb_points, comb_points,
                           kernel=self.kernel,
                           kernel_param=self.kernel_param)
        Eg_rkhs = rkhs_func_exp(w, comb_points, samples,
                                kernel=self.kernel,
                                kernel_param=self.kernel_param)
        g_norm = rkhs_norm(w, comb_points,
                           kernel=self.kernel,
                           kernel_param=self.kernel_param)

        self.constraints = self.x_constraints(x)
        self.constraints.extend([
            f_const <= (1 - mu[1, :]) * M,
            f_const >= -(1 - mu[0, :]) * M,
            g0 + g_rkhs >= mu[0, :],
            g0 + Eg_rkhs + epsilon * g_norm <= alpha,
            mu.T @ np.ones(2) == 1,
        ])
        prob = cp.Problem(objective=cp.Maximize(self.objective(x)),
                          constraints=self.constraints)
        prob.solve(solver="MOSEK")
        if prob.status == 'infeasible':
            raise ValueError

        return prob.value, x.value


class CVaRRelaxation(CCP):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x, kernel_param=None,
                 rkhs_method='representer'):
        super(CVaRRelaxation, self).__init__(objective=objective,
                                             x_constraints=x_constraints,
                                             chance_constraints=chance_constraints,
                                             dim_x=dim_x)
        self.kernel_param = kernel_param
        self.trained = False
        self._w = None
        self._xi_train = None
        self.rkhs_method = rkhs_method

    def solve_problem(self, samples, alpha, epsilon=None, test_xi=None,
                      n_rand_feat=200):
        self.trained = False
        n_samples, dim_xi = samples.shape

        # if necessary compute ambiguity set radius
        if epsilon == 'bootstrap':
            epsilon = compute_bootstrap_rkhs_radius(samples,
                                                    kernel_param=self.kernel_param,
                                                    confidence_level=0.95,
                                                    bootstrap_samples=200)
        elif epsilon == 'rate':
            epsilon = mmd_eps(n_sample=n_samples,
                              alpha=alpha)

        if self.rkhs_method == 'representer':
            # Precompute kernel matrix
            kernel_matrix = compute_gram_matrix(samples, param=self.kernel_param)
            kernel_cholesky = cholesky_decomposition(kernel_matrix)
            w = cp.Variable(shape=(n_samples, 1), name='weights')
            g_rkhs = kernel_matrix @ w
            g_norm = cp.norm(kernel_cholesky @ w)
        elif self.rkhs_method == "rand_feature":
            _, kernel_gamma = median_heuristic(samples, samples)
            rbf_feature = RBFSampler(gamma=kernel_gamma, n_components=n_rand_feat, random_state=1)
            x_feat = rbf_feature.fit_transform(samples)
            w = cp.Variable(shape=(n_rand_feat, 1), name='weights')
            g_norm = cp.norm(w)
            g_rkhs = x_feat @ w
        else:
            raise ValueError('Option not available.')

        # Define variables
        x = cp.Variable(shape=(self.dim_x, 1), name='x')
        t = cp.Variable(1, name='CVaR')
        g0 = cp.Variable(1, name='g0')

        # Define constraints
        Eg_rkhs = cp.sum(g_rkhs) / n_samples
        f_const = self.chance_constraint(x, samples)

        self.constraints = self.x_constraints(x)
        self.constraints.extend([
            g0 + Eg_rkhs + epsilon * g_norm <= t * alpha,
            cp.pos(f_const + t) <= g0 + g_rkhs
        ])
        prob = cp.Problem(objective=cp.Maximize(self.objective(x)),
                          constraints=self.constraints)
        try:
            prob.solve(solver="MOSEK")
            if prob.status in ['infeasible', 'unbounded']:
                print("Infeasible for n_samples: ", n_samples)
                return None, None, None, None, None
            self.trained = True
            self._w = w.value
            self._xi_train = samples
        except Exception as e:
            print(e)
            print("Infeasible for n_samples: ", n_samples)
            return None, None, None, None, None

        obj_sol = prob.value
        x_sol = x.value
        weights = w.value
        g_offset = g0.value

        # compute CVaR
        emp_cvar = self.compute_cvar(test_xi, x_sol, alpha)

        return obj_sol, x_sol, emp_cvar, weights, g_offset

    def compute_cvar(self, test_samples, x_sol, alpha):
        """
        Given the solution of the problem compute the empirical CVaR.

        Parameters
        ----------
        test_samples: ndarray
            Test sample to evaluate empirical CVaR
        x_sol: ndarray
            Solution to DRCCP
        alpha: float
            Risk level of DRCCP

        Returns
        -------
        emp_cvar: float
            Empirical CVaR
        """
        # compute CVaR
        t = cp.Variable(1, 'CVaR')
        f_cvar = self.chance_constraint(x_sol, test_samples).value
        cvar_obj = cp.sum(cp.maximum(f_cvar + t, 0)) / test_samples.shape[0] - t * alpha
        prob = cp.Problem(objective=cp.Minimize(cvar_obj))
        try:
            prob.solve(solver="MOSEK")
            emp_cvar = prob.value
            return emp_cvar
        except:
            print("Could not compute CVaR")

    def eval_rkhs_function(self, xi):
        assert self.trained, 'Problem needs to be solved before RKHS function can be evaluated.'
        return rkhs_func(self._w, self._xi_train, xi, self.kernel_param)

    def eval_rkhs_norm(self):
        assert self.trained, 'Problem needs to be solved before RKHS function can be evaluated.'
        return rkhs_norm(self._w, self._xi_train)


class SOSCVaRRelaxation(CVaRRelaxation):
    def __init__(self, objective, x_constraints, chance_constraints, dim_x,
                 xi_sym, x_sym, kernel_param=None):
        """
        Create an SOS variant of the CVaR relaxation.

        Parameters
        ----------
        objective:
        x_constraints: list
            List with deterministic cvx constraint on the decision variable
        chance_constraints: sympy function
            Constraint function defined in sympy
        dim_x: int
        xi_sym: sympy.Symbols
            Symbol of one instance of RV xi
        x_sym: sympy.Symbols
            Symbol of decision variable X
        var_dict: dict
            Mapping of sympy variables to cvx variables
        kernel_param: dict
        """
        super(SOSCVaRRelaxation, self).__init__(objective=objective,
                                                x_constraints=x_constraints,
                                                chance_constraints=chance_constraints,
                                                dim_x=dim_x,
                                                kernel_param=kernel_param,
                                                rkhs_method='representer')
        self.xi_sym = xi_sym
        self.xi_arr = np.array([xi for xi in xi_sym])
        self.x_sym = x_sym
        self.x_arr = np.array([x for x in x_sym])
        self.x_var = cp.Variable(dim_x, name='x')
        self.var_dict = {self.x_arr[i]: self.x_var[i] for i in range(dim_x)}
        self.cc_callable = sp.lambdify((self.x_sym, self.xi_sym), self.chance_constraint)

    def eval_cc(self, x, samples):
        """
        Evaluate chance constraint function.

        Parameters
        ----------
        x: ndarray
            One instance of decision variable -- shape = (dim, 1)
        samples: ndarray
            Shape = (n_samples, dim_xi)

        Returns
        -------
        f_value: ndarray
        """
        return np.vstack([self.cc_callable(x, samples[i, :]) for i in range(samples.shape[0])])

    def _setup_problem(self, samples):
        """
        Define necessary variables and symbols for the SOS decomposition and compute
        expression for SOS decomposition.

        Returns
        -------
        lhs1, lhs2: sympy.Poly
            Left-hand side of the SOS constraint.
        """
        n_samples, dim_xi = samples.shape
        k_func = lambda xi1, xi2: (1 + (xi1 @ xi2) ** (self.kernel_param['degree']))
        K = k_func(samples, self.xi_arr)
        gamma_cvx = cp.Variable(shape=(n_samples, 1), name='gamma')
        gamma_sym = sp.MatrixSymbol('gamma', n_samples, 1)
        self.var_dict[gamma_sym] = gamma_cvx
        g_sym = (K @ gamma_sym)[0]
        g0_cvx = cp.Variable(name='g0')
        g0_sym = sp.Symbol('g0')
        self.var_dict[g0_sym] = g0_cvx
        t_cvx = cp.Variable(name='t')
        t_sym = sp.Symbol('t')
        self.var_dict[t_sym] = t_cvx

        lhs1 = g0_sym + g_sym - t_sym - self.chance_constraint
        lhs2 = g0_sym + g_sym

        return lhs1, lhs2, g0_sym, gamma_sym, t_sym

    def solve_problem(self, samples, alpha, epsilon=None, test_xi=None):
        self.trained = False
        n_samples, dim_xi = samples.shape
        lhs1, lhs2, g0_sym, gamma_sym, t_sym = self._setup_problem(samples)

        # todo: fix to work with arbitrary kernel function
        # if necessary compute ambiguity set radius
        if epsilon == 'bootstrap':
            epsilon = compute_bootstrap_rkhs_radius(samples,
                                                    kernel_param=self.kernel_param,
                                                    confidence_level=0.95,
                                                    bootstrap_samples=200)
            print('Bootstrap for {0}: {1}'.format(samples.shape[0], epsilon))
        elif epsilon == 'rate':
            epsilon = mmd_eps(n_sample=n_samples,
                              alpha=alpha)
            print('Rate for {0}: {1}'.format(samples.shape[0], epsilon))
        else:
            print('Epsilon = {}'.format(epsilon))

        if self.rkhs_method == 'representer':
            # Precompute kernel matrix
            kernel_matrix = compute_gram_matrix(samples, param=self.kernel_param)
            kernel_cholesky = cholesky_decomposition(kernel_matrix)
            g_rkhs = kernel_matrix @ self.var_dict[gamma_sym]
            g_norm = cp.norm(kernel_cholesky @ self.var_dict[gamma_sym])
            Eg_rkhs = cp.sum(g_rkhs) / n_samples
            g0 = self.var_dict[g0_sym]
        else:
            raise ValueError('{} is not accepted as parameter.'.format(self.rkhs_method))

        self.constraints = self.x_constraints(self.x_var)
        self.constraints.append(g0 + Eg_rkhs + epsilon * g_norm <= self.var_dict[t_sym]*alpha)

        # SOS decomposition constraints
        sos_const1, Q1 = create_sos_constraints(lhs_constraint=lhs1,
                                                var_sym=self.xi_sym,
                                                var_arr=self.xi_arr,
                                                var_dict=self.var_dict)
        self.constraints.extend(sos_const1)
        sos_const2, Q2 = create_sos_constraints(lhs_constraint=lhs2,
                                                var_sym=self.xi_sym,
                                                var_arr=self.xi_arr,
                                                var_dict=self.var_dict)
        self.constraints.extend(sos_const2)

        prob = cp.Problem(objective=cp.Maximize(self.objective(self.x_var)),
                          constraints=self.constraints)
        try:
            prob.solve(solver="MOSEK")
            if prob.status in ['infeasible', 'unbounded']:
                print("Infeasible for n_samples: ", n_samples)
                return None, None, None, None, None
            self.trained = True
            self._w = self.var_dict[gamma_sym].value
            self._xi_train = samples
        except Exception as e:
            print(e)
            print("Infeasible for n_samples: ", n_samples)
            return None, None, None, None, None

        obj_sol = prob.value
        x_sol = self.x_var.value
        weights = self.var_dict[gamma_sym].value
        g0 = self.var_dict[g0_sym].value
        g_rkhs = g_rkhs.value
        Eg_rkhs = Eg_rkhs.value
        # f_val = self.eval_cc(x_sol, samples)
        t_val = self.var_dict[t_sym].value
        g_norm = g_norm.value

        print('RKHS norm: {}'.format(g_norm))
        print('Slack var in CVaR: {}'.format(t_val))
        print('Emp Exp of g: {}'.format(Eg_rkhs))
        print('G weights: {}'.format(weights[:10]))
        print('Grammian: {}'.format(kernel_matrix[:10, :10]))

        # compute CVaR
        emp_cvar = self.compute_cvar(test_xi, x_sol, alpha)

        return obj_sol, x_sol, emp_cvar, weights, g0

    def compute_cvar(self, test_samples, x_sol, alpha):
        """
        Given the solution of the problem compute the empirical CVaR.

        Parameters
        ----------
        test_samples: ndarray
            Test sample to evaluate empirical CVaR
        x_sol: ndarray
            Solution to DRCCP
        alpha: float
            Risk level of DRCCP

        Returns
        -------
        emp_cvar: float
            Empirical CVaR
        """
        # compute CVaR
        t = cp.Variable(1, 'CVaR')
        f_cvar = self.eval_cc(x_sol, test_samples)
        cvar_obj = cp.sum(cp.maximum(f_cvar + t, 0)) / test_samples.shape[0] - t * alpha
        prob = cp.Problem(objective=cp.Minimize(cvar_obj))
        try:
            prob.solve(solver="MOSEK")
            emp_cvar = prob.value
            return emp_cvar
        except:
            print("Could not compute CVaR")

def cvx_exact_plot(x, chance_constraint, samples, alpha, epsilon, kernel):
    n_samples, dim_xi = samples.shape

    w = cp.Variable(shape=(n_samples, 1), name='weights')
    g0 = cp.Variable(1, name='g0')

    # Precompute kernel matrix
    kernel_matrix = compute_gram_matrix(samples, param={'kernel': kernel})
    kernel_cholesky = cholesky_decomposition(kernel_matrix)

    # Define constraints and objective
    g_rkhs = kernel_matrix @ w
    Eg_rkhs = 1/n_samples * cp.sum(g_rkhs[:n_samples])
    g_norm = cp.norm(kernel_cholesky @ w)
    f_const = chance_constraint(x, samples).value
    ind_func = np.asarray(f_const > 0, dtype=float)
    objective = g0 + Eg_rkhs + epsilon * g_norm

    constraints = [
        ind_func <= g0 + g_rkhs
    ]
    prob = cp.Problem(objective=cp.Minimize(objective),
                      constraints=constraints)

    try:
        prob.solve(solver="MOSEK")
        if prob.status in ['infeasible', 'unbounded']:
            print('Failed')
            return None, None
        else:
            return w.value, g0.value
    except:
        return None, None


def dataset_figure(f_constraint, X, cert_points, x, kernel, kernel_param, alpha, epsilon):
    """
    Create the figure to visualize the effects of different ways to create dataset.

    Parameters
    ----------
    f_constraint
    X
    cert_points
    x
    sigma
    alpha
    epsilon

    Returns
    -------

    """
    if np.any(x < 0):
        return None
    n_samples, dim = X.shape
    comb_points = np.vstack((cert_points, X))
    n_cert = cert_points.shape[0]
    # Define variables
    coef = cp.Variable(n_samples + n_cert, name='coefficients')
    coef.value = np.zeros(n_samples + n_cert)

    # Define constraints
    f_const = np.around(f_constraint(x, comb_points), 6)  # This does not depend on decision variable
    ind_func = np.asarray(f_const > 0, dtype=float)
    g_rkhs = rkhs_func(coef, comb_points, comb_points, kernel=kernel, kernel_param=kernel_param)
    Eg_rkhs = rkhs_func_exp(coef, comb_points, X, kernel=kernel, kernel_param=kernel_param)
    g_norm = rkhs_norm(coef, comb_points, kernel=kernel, kernel_param=kernel_param)
    constraints = [
        Eg_rkhs + epsilon * g_norm <= alpha,
        g_rkhs >= ind_func
    ]

    # Create objective
    obj_fit = cp.Minimize(cp.norm(g_rkhs - ind_func, 2))
    obj_norm = cp.Minimize(Eg_rkhs + epsilon * g_norm)

    if np.any(ind_func == 1):
        # solve twice:
        prob = cp.Problem(obj_fit, constraints)
        prob.solve(solver="MOSEK")
        if prob.status == "infeasible":
            return None
        else:
            g1, g_norm1 = g_rkhs.value, g_norm.value
        prob = cp.Problem(obj_norm, constraints)
        prob.solve(solver="MOSEK")
        g2, g_norm2 = g_rkhs.value, g_norm.value
        print("RKHS-norm comparison: fit -- {0}    norm -- {1}".format(g_norm1, g_norm2))
        import matplotlib.pyplot as plt
        plt.rc('text', usetex=True)
        plt.style.use('../plt_style/jz.mplstyle')
        pts = comb_points[:, 0]
        n_points = 10
        plt.plot(pts[:n_points], ind_func[:n_points], label=r"$1(f(x, \xi) \geq 0$")
        plt.plot(pts[:n_points], g1[:n_points], label="Fit")
        plt.plot(pts[:n_points], g2[:n_points], label="Norm")
        plt.legend()
        plt.show()
        plt.close()


