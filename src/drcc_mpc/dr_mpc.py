import casadi as cas
import numpy as np

from drcc_mpc.st_mpc import TubeMPC
from drccp_utils.rkhs_utils import (median_heuristic, rbf_kernel, linear_kernel,
                                    compute_bootstrap_rkhs_radius)
from drccp_utils.casadi_utils import (cas_rkhs_norm_squared,
                                      cas_rkhs_func_exp,
                                      cas_rkhs_func)


def ellipsoid_constraint(center, axis):
    """
    Create ellipsoid constraint.

    Parameters
    ----------
    center: ndarray
    axis: ndarray

    Returns
    -------
    ellipsoid_c: casadi.Function
    """
    dim_x = len(center)
    z = cas.MX.sym('z', dim_x)
    e = cas.MX.sym('e', dim_x)
    tmp_lst = []
    for i in range(dim_x):
        tmp_lst.append(((z[i] + e[i] - center[i])/axis[i])**2)
    ellipsoid_c = cas.Function('e_c', [z, e], [sum(tmp_lst) - 1])
    return ellipsoid_c


def exponential_constraint(x_lvl, growth_rate):
    """
    Exponential constraint for 2D system.

    Parameters
    ----------
    x_lvl: float
    growth_rate: float

    Returns
    -------
    exp_c: casadi.Function
    """
    z = cas.MX.sym('z', 2)
    e = cas.MX.sym('e', 2)
    exp_c = cas.Function('exp_c', [z, e],
                         [-1 + x_lvl + cas.exp(growth_rate * (z[0] + e[0])) - z[1] - e[1]])
    return exp_c


class DRCVaRTubeMPC(TubeMPC):
    def __init__(self, dynamics, horizon, task_length, n_scenarios, noise,
                 Q, R, state_constraint, input_constraint, alpha, kernel, kernel_param):
        """
        Scenario Tube MPC for LTI systems with additive noise.

        Parameters
        ----------
        dynamics: smpc.dynamics.LTIDynamics
        horizon: int
        n_scenarios: int
        noise: BaseNoise
        Q: ndarray
        R: ndarray
        state_constraint: callable
        input_constraint: callable
        alpha: float
            Risk level of chance constraint
        epsilon: float
            Regularization due to DRO formulation
        """
        super(DRCVaRTubeMPC, self).__init__(dynamics=dynamics,
                                            horizon=horizon,
                                            task_length=task_length,
                                            n_scenarios=n_scenarios,
                                            noise=noise,
                                            Q=Q, R=R)
        # Define additional CVaR related decision variables
        self.t = self.opti.variable(1, self.horizon)
        self.g0 = self.opti.variable(1, self.horizon)
        self.w = self.opti.variable(self.n_samples, self.horizon)
        self.s = self.opti.variable(1, self.horizon)
        # Constraint functions
        self.state_constraint = state_constraint
        self.input_constraint = input_constraint

        self.alpha = alpha
        if kernel == 'rbf':
            self.kernel = rbf_kernel
        elif kernel == 'linear':
            self.kernel = linear_kernel
        else:
            raise ValueError
        self.kernel_param = kernel_param

        # Final setup call to initialize tube
        self.tube.setup(self.task_length, self.n_samples)
        self.tube.fit_tube(0)

    def setup_mpc(self, horizon, n_samples, **kwargs):
        self.horizon = horizon
        self.n_samples = n_samples
        self.tube.setup(self.task_length, self.n_samples)
        self.tube.fit_tube(0)
        self.trajectory.reset()
        self.alpha = kwargs['alpha']
        # compute epsilon for every iteration with bootstrap
        conf_lvl = kwargs.get('conf_lvl', 0.99)
        self.epsilons = []
        for i in range(self.tube.task_len):
            self.epsilons.append(compute_bootstrap_rkhs_radius(self.tube.scenarios[i+1, :, :],
                                                               kernel_param=self.kernel_param,
                                                               confidence_level=conf_lvl))

    def initialize(self, initial_state):
        self.trajectory.reset()
        self.nominal_state = initial_state
        self.current_state = initial_state

    def remove_constraints(self):
        self.opti.subject_to()

    def _compute_objective(self):
        """
        Compute objective as sum over disturbances.

        Returns
        -------
        cost: TBD
        """
        cost = 0
        for i in range(self.n_samples):
            x = self.z0 + self.e_tensor[0][0, :].T
            u = self.V[:, 0] + self.tube.K @ self.e_tensor[0][0, :].T
            cost += x.T @ self.Q @ x + u.T @ self.R @ u
            for j in range(self.horizon-1):
                x = self.Z[:, j] + self.e_tensor[j][i, :].T
                u = self.V[:, j+1] + self.tube.K @ self.e_tensor[j][i, :].T
                cost += x.T @ self.Q @ x + u.T @ self.R @ u

            # final cost
            x_N = self.Z[:, self.horizon-1] + self.e_tensor[self.horizon-1][i, :].T
            cost += x_N.T @ self.Q @ x_N
        return cost

    def add_nominal_dynamics_constraints(self):
        for j in range(self.horizon):
            if j == 0:
                self.opti.subject_to(
                    self.Z[:, j] == self.nom_dyn(self.z0, self.V[:, j]))
            else:
                self.opti.subject_to(
                    self.Z[:, j] == self.nom_dyn(self.Z[:, j-1], self.V[:, j]))

    def add_state_constraints(self, idx):
        pred_horizon = min(self.horizon, self.task_length - idx)
        for i in range(pred_horizon):
            eval_points = self.tube.scenarios[idx + 1 + i, :, :]
            # f_val = self.state_constraint.eval(self.Z[:, i], eval_points)

            f_vals = []
            for j in range(eval_points.shape[0]):
                f_vals.append(self.state_constraint(self.Z[:, i], eval_points[j, :]))
            f_val = cas.vcat(f_vals)

            # Compute rkhs related values
            sigma, _ = median_heuristic(eval_points, eval_points)

            K = self.kernel(eval_points, eval_points, sigma)
            g_rkhs = cas_rkhs_func(self.g0[:, i], self.w[:, i], K)
            Eg_rkhs = cas_rkhs_func_exp(self.g0[:, i], self.w[:, i], K)
            g_norm_squared = cas_rkhs_norm_squared(self.w[:, i], K)

            self.opti.subject_to(self.g0[:, i] + Eg_rkhs + self.epsilons[i] * self.s[:, i] <= self.t[:, i] * self.alpha)
            self.opti.subject_to(f_val + self.t[:, i] <= self.g0[:, i] + g_rkhs)
            self.opti.subject_to(0 <= self.g0[:, i] + g_rkhs)
            self.opti.subject_to(g_norm_squared <= self.s[:, i]**2)
            self.opti.subject_to(0 <= self.s[:, i])

    def solve_mpc_iteration(self, current_state, idx):
        self.remove_constraints()

        # Set initial state
        e0 = current_state - self.nominal_state
        self.trajectory.add(nominal=self.nominal_state, error=e0)
        self.current_state = current_state
        self.opti.set_value(self.z0, self.nominal_state)

        pred_horizon = min(self.horizon, self.task_length - idx)
        for i in range(pred_horizon):
            self.opti.set_value(self.e_tensor[i],
                                self.tube.scenarios[idx + 1, :, :])
        for i in range(pred_horizon, self.horizon):
            self.opti.set_value(self.e_tensor[i],
                                np.zeros((self.n_samples, self.x_dim)))

        # Update constraints
        self.add_nominal_dynamics_constraints()
        self.add_state_constraints(idx)

    # Solve system
        try:
            sol = self.opti.solve()
        except:
            raise ValueError
            print('Failed')

        # Update nominal state for next iteration
        self.nominal_predictions[:, 0] = sol.value(self.z0)
        self.nominal_predictions[:, 1:] = sol.value(self.Z)[:, :-1]
        self.nominal_state = sol.value(self.Z[:, 0])

        # Add tube/feedback term to action.
        action = sol.value(self.V[0]) + self.tube.K @ e0
        self.trajectory.add_action(action)
        return action

    def add_input_constraints(self, idx):
        return
        pred_horizon = min(self.horizon, self.task_length - idx)
        for i in range(pred_horizon):
            pass

    def plot_trajectory(self, constraint, fig=None, ax=None, **kwargs):
        assert self.x_dim == 2, "Can only plot trajectory fro 2d systems."
        import matplotlib.pyplot as plt
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        zorder = kwargs.get('zorder', 1)
        self.trajectory.plot_trajectory(ax, color='k', alpha=1, label='traj', zorder=zorder)
        if constraint is not None:
            x = np.arange(-5, 12, 0.1)
            x = np.vstack((x, np.zeros_like(x))).T
            e = np.array([0, 0])
            y = constraint(e, x)
            ax.plot(x[:, 0], y, 'r')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')

    def add_terminal_constraint(self, idx):
        pass

    def save_run(self, folder, name):
        self.trajectory.save_trajectory(folder, 'traj_' + name)
        self.trajectory.reset()
        self.tube.save_scenarios(folder, 'err_' + name)
