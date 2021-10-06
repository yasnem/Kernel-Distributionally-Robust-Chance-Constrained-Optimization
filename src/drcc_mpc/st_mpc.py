import argparse
import numpy as np
import matplotlib.pyplot as plt
from casadi import (Opti, hcat, vcat, sumsqr, mtimes, Function, MX, vec,
                    norm_2, exp)
from abc import ABCMeta, abstractmethod
from sklearn.metrics.pairwise import rbf_kernel

from drcc_mpc.constraints import svm_upperbound, compute_f, compute_rkhsnorm_squared
from drcc_mpc.dynamics import LTIDynamics, NormalNoise, Trajectory
from drcc_mpc.error_dynamics import PRSTube, BaseTube
from drcc_mpc.constraint_sets import Box
from drccp_utils.utils import get_project_root



class MPC(metaclass=ABCMeta):

    @abstractmethod
    def initialize(self, initial_state):
        pass

    @abstractmethod
    def setup_mpc(self, horizon, n_samples, **kwargs):
        pass

    @abstractmethod
    def solve_mpc_iteration(self, current_state, idx):
        pass

    # def save(self, save_dir, name):
    #     pass
    #
    # def load(self, save_dir, name):
    #     pass


class sMPC(MPC):
    """
    Scenario Approach for stochastic mpc assuming lti systems with additive
    gaussian noise.

        x_{t+1} = A*x_t + B*u_t + w_t
            __
    Stage cost: l(x, u) = x^T*Q*x + u^T*R*u

    """
    def __init__(self, dynamics, horizon, n_scenarios, noise,
                 Q, R, state_constraint, input_constraint, feedback):
        """
        Set up scenario approach for MPC.

        Parameters
        ----------
        dynamics: smpc.dynamics.LTIDynamics
        horizon: int
            Number of steps to evolve system
        n_scenarios: int
            Number of scenarios
        noise: callable
        Q: ndarray
            Quadratic state cost
        R: ndarray
            Quadratic input cost
        state_constraint: smpc.probabilistic_reachable_sets.Box
        input_constraint: smpc.probabilistic_reachable_sets.Box
        feedback: str
            Feedback type: ['none', 'disturbance']
        """
        self.opti = Opti()
        self.opti.solver('ipopt', {'print_time': False}, {'print_level':0})

        # Collection of all decision variables
        self.x_dim = dynamics.A.shape[1]
        self.u_dim = dynamics.B.shape[1]
        self.noise_dim = dynamics.D.shape[1]
        self.noise = noise
        self.n_samples = n_scenarios
        self.horizon = horizon
        self.X = []
        self.U = []
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.X_constraint = state_constraint
        self.U_constraint = input_constraint
        self.feedback = feedback
        self.trajectory = Trajectory()
        self.current_state = np.zeros(self.x_dim)

    def reset(self):
        self.trajectory.reset()

    def initialize(self, initial_state):
        self.current_state = initial_state
        self.trajectory.add(state=self.current_state)

    def setup_mpc(self, horizon, n_samples, **kwargs):
        self.horizon = horizon
        self.n_samples = n_samples
        self._setup_system()
        objective = self._compute_objective()
        self.opti.minimize(objective)

    def _compute_objective(self):
        """Create objective."""
        expected_cost = 0
        for k in range(self.n_samples):
            expected_cost += sumsqr(self.X[k])
        expected_cost /= self.n_samples
        expected_cost += sumsqr(self.U)
        return expected_cost

    def _disturbance_feedback(self):
        """
        Create action parametrization according to disturbance feedback.
        """
        gamma = self.opti.variable(self.u_dim, self.horizon)
        thetas = [[] for i in range(self.horizon)]
        for i in range(self.horizon):
            for j in range(i):
                thetas[i].append(self.opti.variable(self.u_dim,
                                                    self.noise_dim))

        for i in range(self.n_samples):
            for j in range(self.horizon):
                u = gamma[:, j]
                for k in range(j):
                    u += mtimes(thetas[j][k],
                                self.w[i][:, k])
                self.U.append(u)
        self.U = hcat(self.U)

    def _feedforward(self):
        """
        Feedforward feedback creation.
        """
        self.U = self.opti.variable(self.u_dim, self.horizon)

    def _setup_system(self):
        """
        Set up decision variables.

        Parameters
        ----------
        initial_state: ndarray
        feedback: str
            Feedback type:
                None -- No feedback
                disturbance -- u = a + \sum_{i=k}^{Horizon} K * w
        """
        # Set initial state as parameter to be set when solving mpc iteration
        self.x0 = self.opti.parameter(self.x_dim)
        self.w = [self.opti.parameter(self.noise_dim, self.horizon) for k in
                  range(self.n_samples)]
        self.X = [self.opti.variable(self.x_dim, self.horizon) for k in
                  range(self.n_samples)]
        self.U = []
        self.dyn = self.dynamics.dyn

        # Create control variables
        if self.feedback == 'none':
            self._feedforward()
        elif self.feedback == 'disturbance':
            self._disturbance_feedback()
        else:
            raise NotImplementedError
        # First remove all previous constraints
        self.opti.subject_to()
        self.add_dynamic_constraints()
        self.add_input_state_constraints()

    def add_dynamic_constraints(self):
        for i in range(self.n_samples):
            for j in range(self.horizon):
                if j == 0:
                    self.opti.subject_to(
                        self.X[i][:, j] == self.dyn(self.x0,
                                                    self.U[:, j],
                                                    self.w[i][:, j]))
                else:
                    self.opti.subject_to(
                        self.X[i][:, j] == self.dyn(self.X[i][:, j - 1],
                                                    self.U[:, j],
                                                    self.w[i][:, j]))

    def add_input_state_constraints(self):
        # input constraints
        self.opti.subject_to(self.opti.bounded(self.U_constraint.lb,
                                               self.U,
                                               self.U_constraint.ub))

        for i in range(self.n_samples):
            # state constraints
            self.opti.subject_to(self.opti.bounded(self.X_constraint.lb,
                                                   self.X[i],
                                                   self.X_constraint.ub))

    def solve_mpc_iteration(self, current_state, idx):
        """Solve optimization for desired initial states and return action."""
        # Set current state parameter
        self.current_state = current_state
        self.trajectory.add(state=current_state)
        self.opti.set_value(self.x0, current_state)
        # Sample noise and set the parameters
        for i in range(self.n_samples):
            w = self.noise(self.horizon)
            self.opti.set_value(self.w[i], w)

        sol = self.opti.solve()
        action = np.atleast_1d(sol.value(self.U[0]))
        self.trajectory.add_action(action)
        return action

    def plot_trajectory(self, constraint=None):
        assert self.x_dim == 2, "Can only plot trajectory fro 2d systems."
        import matplotlib.pyplot as plt
        root = get_project_root()
        plt.style.use(root / 'plt_style/jz.mplstyle')
        fig, ax = plt.subplots()
        self.trajectory.plot_trajectory(ax)
        if constraint is not None:
            x = np.arange(-5, 12, 0.1)
            x = np.vstack((x, np.zeros_like(x))).T
            e = np.array([0, 0])
            y = constraint(e, x)
            ax.plot(x[:, 0], y, 'r')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.show()
        plt.close()

    # These two methods are not meant to be used for the scenario approach
    def save(self, save_dir, name):
        raise NotImplementedError

    def load(self, save_dir, name):
        raise NotImplementedError


class TubeMPC(MPC, metaclass=ABCMeta):
    def __init__(self, dynamics, horizon, task_length, n_scenarios, noise,
                 Q, R):
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
        """

        # Collection of all decision variables
        self.noise = noise
        self.n_samples = n_scenarios
        self.horizon = horizon
        self.task_length = task_length
        self.Z = None
        self.V = None
        self.x_dim = dynamics.A.shape[1]
        self.u_dim = dynamics.B.shape[1]
        self.noise_dim = dynamics.D.shape[1]

        # Dynamics
        self.A = dynamics.A
        self.B = dynamics.B
        self.Q = Q
        self.R = R

        # Logging
        self.nominal_state = np.zeros(self.x_dim)
        self.current_state = np.zeros(self.x_dim)
        self.trajectory = Trajectory()
        self.nominal_predictions = np.zeros((self.x_dim, self.horizon))

        self.tube = BaseTube(self.A, self.B, Q, R, task_length,
                             n_scenarios, noise)
        self._setup_system()

    def _setup_system(self):
        """
        Set up decision variables with feedforward actions for nominal system.
        """

        # Set initial state as parameter to be set when solving mpc iteration
        self.opti = Opti()
        self.opti.solver('ipopt', {'print_time': False},
                         {'print_level': 0,
                          'max_iter': 5000})
        self.z0 = self.opti.parameter(self.x_dim)
        self.Z = self.opti.variable(self.x_dim, self.horizon)
        self.V = self.opti.variable(self.u_dim, self.horizon)
        self.z = MX.sym('z', self.x_dim)
        self.v = MX.sym('v', self.u_dim)
        self.e_tensor = [self.opti.parameter(self.n_samples, self.x_dim) for i in range(self.horizon)]
        z_new = mtimes(self.A, self.z) + mtimes(self.B, self.v)
        self.nom_dyn = Function('f_z', [self.z, self.v], [z_new])
        objective = self.compute_objective()
        self.opti.minimize(objective)

    def initialize(self, initial_state):
        self.nominal_state = initial_state
        self.current_state = initial_state

    def remove_constraints(self):
        self.opti.subject_to()

    def compute_objective(self):
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

    @abstractmethod
    def add_state_constraints(self, idx):
        pass

    @abstractmethod
    def add_input_constraints(self, idx):
        pass

    @abstractmethod
    def add_terminal_constraint(self, idx):
        pass

    def save_run(self, folder, name):
        self.trajectory.save_trajectory(folder, 'traj_' + name)
        self.trajectory.reset()
        self.tube.save_scenarios(folder, 'err_' + name)


class sTubeMPC(TubeMPC):
    def __init__(self, dynamics, horizon, task_length, n_scenarios, noise,
                 Q, R, state_constraint, input_constraint):
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
        state_constraint: smpc.probabilistic_reachable_sets.Box
        input_constraint: smpc.probabilistic_reachable_sets.Box
        """
        super(sTubeMPC, self).__init__(dynamics=dynamics,
                                       horizon=horizon,
                                       task_length=task_length,
                                       n_scenarios=n_scenarios,
                                       noise=noise,
                                       Q=Q, R=R)
        self.tight_X_sets = []
        self.tight_U_sets = []
        self.X_constraint = state_constraint
        self.c_func = state_constraint.get_casadi_function()
        self.U_constraint = input_constraint
        self.g_func = input_constraint.get_casadi_function()
        self.tube = PRSTube(A=dynamics.A,
                            B=dynamics.B,
                            Q=Q, R=R,
                            task_len=task_length,
                            n_samples=n_scenarios,
                            noise=noise)

    def setup_mpc(self, horizon, n_samples, **kwargs):
        self.remove_constraints()
        self.horizon = horizon
        self.nominal_predictions = np.zeros((self.x_dim, self.horizon))
        self.n_samples = n_samples
        self.task_length = kwargs['task_length']
        # self._setup_system()
        self.tube.setup(self.task_length, n_samples)
        self.offline_prs_computation(self.task_length)
        self.trajectory.reset()

    def add_state_constraints(self, idx):
        """Add tightened state constraints to system."""
        for i, prs in enumerate(self.tight_X_sets[idx:idx + self.horizon]):
            self.opti.subject_to(self.opti.bounded(prs.lb,
                                                   self.Z[:, i],
                                                   prs.ub))

    def add_input_constraints(self, idx):
        """Add tightened input constraints to system."""
        for i, prs in enumerate(self.tight_U_sets[idx:idx + self.horizon]):
            self.opti.subject_to(self.opti.bounded(prs.lb,
                                                   self.V[:, i],
                                                   prs.ub))

    def add_terminal_constraint(self, idx):
        """Add terminal state constraint."""
        set_list = self.tight_X_sets[idx:idx + self.horizon]
        terminal_set = Box(set_list[0].n, set_list[0].lb, set_list[0].ub)
        for prs in set_list:
            terminal_set.intersection(prs)
        self.opti.subject_to(self.opti.bounded(terminal_set.lb,
                                               self.Z[:, -1],
                                               terminal_set.ub))

    def offline_prs_computation(self, task_length):
        assert task_length == self.tube.task_len
        self.tube.fit_tube(0)
        self.tight_X_sets.append(self.X_constraint)
        for i in range(task_length):
            s_prs = self.tube.state_prs[i]
            u_prs = self.tube.input_prs[i]
            self.tight_X_sets.append(self.X_constraint.pontryagin_difference(s_prs))
            self.tight_U_sets.append(self.U_constraint.pontryagin_difference(u_prs))

    def solve_mpc_iteration(self, initial_state, idx):
        # Reset constraints and set initial values
        self.remove_constraints()
        e0 = initial_state - self.nominal_state
        # log trajectories
        self.trajectory.add(nominal=self.nominal_state, error=e0)
        self.current_state = initial_state
        self.opti.set_value(self.z0, self.nominal_state)
        pred_horizon = min(self.horizon, self.tube.task_len - idx)
        for i in range(pred_horizon):
            self.opti.set_value(self.e_tensor[i],
                                self.tube.scenarios[idx + 1, :, :])
        for i in range(pred_horizon, self.horizon):
            self.opti.set_value(self.e_tensor[i],
                                np.zeros((self.n_samples, self.x_dim)))

        # Update constraints on nominal states and input
        self.add_nominal_dynamics_constraints()
        self.add_state_constraints(idx)
        self.add_input_constraints(idx)
        self.add_terminal_constraint(idx)

        # Compute objective and solve for open-loop actions
        sol = self.opti.solve()
        # Update nominal state for next iteration
        self.nominal_predictions[:, 0] = sol.value(self.z0)
        self.nominal_predictions[:, 1:] = sol.value(self.Z)[:, :-1]
        self.nominal_state = sol.value(self.Z[:, 0])
        # self.plot_prediction_tube(idx)

        # Add tube/feedback term to action.
        action = sol.value(self.V[0]) + self.tube.K @ e0
        self.trajectory.add_action(action)
        return action

    def plot_prediction_tube(self, idx):
        """Plot tube around prediction at time step idx."""
        assert self.tube.fitted, "Can only plot a tube with generated scenarios."
        fig, ax_arr = plt.subplots(self.tube.dim_x, sharex=True)
        traj = np.asarray(self.error_trajectory) + np.asarray(self.nominal_trajectory)
        nominal_traj = np.hstack((traj[-1:, :].T, self.nominal_predictions))
        scenarios = self.tube.scenarios[1:, :, :]
        max_idx = len(self.tube.state_prs)
        pred_horizon = min(self.horizon, self.tube.task_len - idx)
        for i, ax in enumerate(ax_arr):
            ax.plot(traj[:, i])
            ax.axvline(traj.shape[0]-1, c='grey', lw=1)
            x = np.arange(pred_horizon+1) + traj.shape[0] -1
            ax.plot(x, nominal_traj[i, :pred_horizon+1])
            ub, lb = [0], [0]
            ub.extend([self.tube.state_prs[min(idx+k, max_idx)].ub[i] for k in range(pred_horizon)])
            lb.extend([self.tube.state_prs[min(idx+k, max_idx)].lb[i] for k in range(pred_horizon)])
            ax.fill_between(x, nominal_traj[i, :pred_horizon+1] + lb,
                            nominal_traj[i, :pred_horizon+1] + ub, alpha=0.2)
            for j in range(self.n_samples):
                ax.scatter(x[1:],
                           nominal_traj[i, 1:pred_horizon+1] + scenarios[idx:min(idx+pred_horizon, max_idx), j, i],
                           marker='.', linewidths=1, c='k')
            ax.set_xlabel('timesteps')
            ax.set_ylabel('x(k)')
        plt.show()

    def plot_trajectory(self, constraint=None):
        assert self.x_dim == 2, "Can only plot trajectory fro 2d systems."
        import matplotlib.pyplot as plt
        root = get_project_root()
        plt.style.use(root / 'plt_style/jz.mplstyle')
        fig, ax = plt.subplots()
        self.trajectory.plot_trajectory(ax)
        if constraint is not None:
            x = np.arange(-5, 12, 0.1)
            x = np.vstack((x, np.zeros_like(x))).T
            e = np.array([0, 0])
            y = constraint(e, x)
            ax.plot(x[:, 0], y, 'r')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.show()
        plt.close()


class ConstraintTubeMPC(TubeMPC):
    def __init__(self, dynamics, horizon, task_length, n_scenarios, noise,
                 Q, R, state_constraints, input_constraints, epsilons,
                 method='gradient', **kwargs):
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
        state_constraints: [casadi.Function, ..]
        input_constraints: [casadi.Function, ..]
        epsilon: float
        """
        super(ConstraintTubeMPC, self).__init__(dynamics=dynamics,
                                                horizon=horizon,
                                                task_length=task_length,
                                                n_scenarios=n_scenarios,
                                                noise=noise,
                                                Q=Q, R=R)

        self.epsilons = epsilons
        self.c_func = state_constraints
        self.c_out = [max(func.size_out(0)) for func in self.c_func]
        self.g_func = input_constraints
        self.g_out = [max(func.size_out(0)) for func in self.g_func]
        self.method = method

        if self.method == 'gradient':
            self.c_jac = [func.jacobian() for func in self.c_func]
            self.g_jac = [func.jacobian() for func in self.g_func]
            self.s = self.opti.variable(self.n_samples, np.sum(self.c_out))
            self.t = self.opti.variable(self.n_samples, np.sum(self.c_out))
        elif self.method == 'kdrompc':
            self.a_values=[[] for i in range(self.horizon)]
            self.n_points = kwargs['grid_points']
            self.gamma = kwargs['gamma']
            self.t = self.opti.variable(np.sum(self.c_out), self.horizon)
            self.f0 = [self.opti.variable(np.sum(self.c_out)) for i in range(self.horizon)]
            self.a = [self.opti.variable(np.sum(self.c_out), self.n_samples + self.n_points**2) for i in range(self.horizon)]
            for i in range(self.horizon):
                self.opti.set_initial(self.a[i], 0.1 * np.ones((np.sum(self.c_out), self.n_samples + self.n_points**2)))
                self.opti.set_initial(self.f0[i], 0)
            self.zeta = np.zeros((self.horizon, self.n_points**2, self.x_dim))
        elif self.method == 'kdro_randomfeat':
            self.n_points = 0
            self.gamma = kwargs['gamma']
            self.t = self.opti.variable()
            self.f0 = self.opti.variable(np.sum(self.c_out))
            # self.f0 = 0
            self.a = self.opti.variable(np.sum(self.c_out), self.n_samples + self.n_points**2)
            self.opti.set_initial(self.a, 0.1 * np.ones((np.sum(self.c_out), self.n_samples + self.n_points**2)))
            # self.a = np.zeros((np.sum(self.c_out), self.n_samples))
            self.opti.set_initial(self.f0, 0)
            self.zeta = np.zeros((self.horizon, self.n_points**2, self.x_dim))
        elif self.method == 'svm':
            self.gamma = kwargs['gamma']
            self.svm = kwargs['svm']
        elif self.method == 'stubempc':
            pass
        else:
            raise NotImplementedError

    def _setup_constraints(self):
        e_vec = MX.sym('e_vec', self.n_samples, self.x_dim)
        c_vals = []
        for func in self.c_func:
            tmp = []
            for i in range(self.n_samples):
                tmp.append(func(self.z, e_vec[i, :].T))
            c_vals.append(hcat(tmp))
        self.c_eval_fin = Function('c_vec', [self.z, e_vec],
                                   [vcat(c_vals)])
        self.c_eval = self.c_eval_fin

        g_vals = []
        for func in self.g_func:
            tmp = []
            for i in range(self.n_samples):
                tmp.append(func(self.v, self.tube.K@e_vec[i, :].T))
            g_vals.append(hcat(tmp))
        self.g_eval = Function('g_vec', [self.v, e_vec], [vcat(g_vals)])

        if self.method == 'gradient':
            c_grad = []
            for k, func in enumerate(self.c_func):
                for i in range(self.n_samples):
                    for j in range(self.c_out[k]):
                        c_grad.append(norm_2(self.c_jac[k](self.z, e_vec[i, :],
                                                           func(self.z, e_vec[i, :].T))[j, self.x_dim:]))
            self.c_grad_eval = Function('c_grad', [self.z, e_vec],
                                        [vcat(c_grad)])
            g_grad = []
            for k, func in enumerate(self.g_func):
                for i in range(self.n_samples):
                    for j in range(self.g_out[k]):
                        g_grad.append(norm_2(self.g_jac[k](self.v, self.tube.K@e_vec[i, :].T,
                                                           func(self.v, self.tube.K@e_vec[i, :].T))[j, self.x_dim:]))
            self.g_grad_eval = Function('g_grad', [self.z, e_vec],
                                        [vcat(g_grad)])

        elif self.method == 'kdrompc' or self.method == 'kdro_randomfeat' :
            c_vals = []
            e_zeta = MX.sym('e_zeta', self.n_samples + self.zeta.shape[1], self.x_dim)
            for func in self.c_func:
                tmp = []
                for i in range(e_zeta.shape[0]):
                    tmp.append(func(self.z, e_zeta[i, :]))
                c_vals.append(hcat(tmp))
            self.c_eval = Function('c_vec', [self.z, e_zeta],
                                   [vcat(c_vals)])
        elif self.method == 'svm' or self.method == 'stubempc':
            pass
        else:
            raise NotImplementedError

    def setup_mpc(self, horizon, n_samples, **kwargs):
        self.remove_constraints()
        self.horizon = horizon
        self.n_samples = n_samples
        self.nominal_predictions = np.zeros((self.x_dim, self.horizon))
        self.task_length = kwargs['task_length']
        if 'epsilons' in kwargs:
            self.epsilons = kwargs['epsilons']
        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        self._setup_constraints()
        # self._setup_system()
        self.tube.setup(self.task_length, self.n_samples)
        self.tube.fit_tube(0)
        if self.method == 'kdrompc' or self.method == 'kdro_randomfeat':
            for i in range(self.horizon):
                scen = self.tube.scenarios[i + 1, :, :]
                xx = np.linspace(np.min(scen, axis=0) - 1,
                                 np.max(scen, axis=0) + 1,
                                 num=self.n_points)
                x, y = np.meshgrid(xx[:, 0], xx[:, 1])
                X = np.vstack((x.ravel(), y.ravel())).T
                self.zeta[i, :, :] = X
        self.trajectory.reset()

    def add_state_constraints(self, idx):
        pred_horizon = min(self.horizon, self.task_length - idx)

        for i in range(pred_horizon):
            # for every z step compute the max margin
            if self.method == 'gradient':
                c = self.c_eval(self.Z[:, i], self.e_tensor[i])
                c_grad = self.c_grad_eval(self.Z[:, i], self.e_tensor[i])
                self.opti.subject_to(vec(c) <= vec(self.t))
                self.opti.subject_to(self.epsilons[idx+i] * vec(c_grad) <= vec(self.s))
                self.opti.subject_to(vec(self.s) + vec(self.t) <= 0)
            elif self.method == 'svm':
                c = self.c_eval(self.Z[:, i], self.e_tensor[i])
                rkhs_norm = svm_upperbound(self.svm)
                self.opti.subject_to(vec(c) + self.epsilons[idx+i] * rkhs_norm <= 0)
            elif self.method == 'kdrompc':
                eval_points = np.vstack((self.tube.scenarios[idx + 1 + i, :, :],
                                         self.zeta[i, :, :]))
                c = self.c_eval(self.Z[:, i], eval_points)
                K = rbf_kernel(eval_points, eval_points, gamma=self.gamma)
                f_ze = compute_f(self.a[i], self.f0[i], K)
                f_e = compute_f(self.a[i], self.f0[i], K[:self.n_samples, :])
                rkhs_norm_squared = compute_rkhsnorm_squared(self.a[i], K)
                # def log_alpha(idx):
                #     for i in range(self.horizon):
                #         self.a_values[i].append(np.sqrt(self.opti.value(self.f0[i])))
                # self.opti.callback(log_alpha)
                self.opti.subject_to(vec(c) <= vec(f_ze))
                self.opti.subject_to(rkhs_norm_squared <= self.t[:, i]**2)
                self.opti.subject_to(0 <= self.t[:, i])
                self.opti.subject_to(vec(f_e + self.epsilons[idx+i] * self.t[:, i]) <= 0)

            elif self.method == 'kdro_randomfeat':
                from sklearn.kernel_approximation import RBFSampler

                eval_points = vcat([self.e_tensor[i], self.zeta[i, :, :]])
                c = self.c_eval(self.Z[:, i], eval_points)
                eval_points = np.vstack((self.tube.scenarios[idx + 1 + i, :, :],
                                         self.zeta[i, :, :]))

                ### begin random feature====================
                w_feat = self.opti.variable(50)  # weights for the random features; # random feature
                self.opti.set_initial(w_feat, 0.0001)
                # sklearn implementation
                rbf_feature = RBFSampler(gamma=self.gamma, n_components=50, random_state=1)
                x_feat = rbf_feature.fit_transform(eval_points)

                # x_feat = get_rf(data_emp, width=kernel_width, n_components=n_rand_feat) # use my implementation of rf; bug for now
                f_empirical = x_feat @ w_feat

                z_feat = rbf_feature.fit_transform(eval_points)  # the newly sampled points

                # z_feat = get_rf(zetai, width=kernel_width, n_components=n_rand_feat)
                f_e = z_feat @ w_feat

                # constr by RF
                s=self.opti.variable() # aux variable
                tmax=self.opti.variable() # aux variable
                f0=self.opti.variable()
                rkhs_norm = norm_2(w_feat)
                self.opti.subject_to(rkhs_norm == s)
                # pmoment = sum1(fmax(vec(c) - vec(f_e), 0)) / (0.1 * vec(c).shape[0])
                self.opti.subject_to(vec(f_empirical) <= tmax )
                # self.opti.subject_to(sum1(vec(f_empirical))/vec(c).shape[0] <= self.t ) # average instead of max
                self.opti.subject_to(vec(c) - vec(f_e) <= f0)
                # print(self.epsilons[idx + i])
                self.opti.subject_to(tmax +f0+  self.epsilons[idx + i] * s <= 0)
            elif self.method == 'stubempc':
                c = self.c_eval(self.Z[:, i], self.e_tensor[i])
                self.opti.subject_to(vec(c) <= 0)
            else:
                raise NotImplementedError

    def add_input_constraints(self, idx):
        pred_horizon = min(self.horizon, self.task_length - idx)
        for i in range(pred_horizon):
            # for every control compute the max margin
            g = self.g_eval(self.V[:, i], self.e_tensor[i])
            self.opti.subject_to(vec(g) <= 0)

    def add_terminal_constraint(self, idx):
        pred_horizon = min(self.horizon, self.task_length - idx)
        for i in range(pred_horizon):
            c = self.c_eval_fin(self.Z[:, -1], self.e_tensor[i])
            self.opti.subject_to(vec(c) <= 0)

    def get_values(self, idx):
        i = 0
        if self.method == 'stubempc':
            eval_points = self.tube.scenarios[idx + 1 + i, :, :]
            c = self.c_eval(self.Z[:, i], eval_points)
            return (self.opti.value(self.Z[:, i]), eval_points,
                    self.opti.value(c))
        elif self.method == 'kdrompc':
            eval_points = np.vstack((self.tube.scenarios[idx + 1 + i, :, :],
                                     self.zeta[i, :, :]))
            c = self.c_eval(self.Z[:, i], eval_points)
            K = rbf_kernel(eval_points, eval_points, gamma=self.gamma)
            f_ze = compute_f(self.a[i], self.f0[i], K)
            rkhs_norm_squared = compute_rkhsnorm_squared(self.a[i], K)
            return (self.opti.value(self.a[i]), self.opti.value(K), self.opti.value(f_ze),
                    np.sqrt(self.opti.value(rkhs_norm_squared)), self.opti.value(self.Z[:, i]),
                    eval_points, self.opti.value(c))

    def solve_mpc_iteration(self, initial_state, idx):
        self.a_values = [[] for i in range(self.horizon)]
        # remove constraints and set initial values
        self.remove_constraints()
        e0 = initial_state - self.nominal_state
        # log trajectories
        self.trajectory.add(nominal=self.nominal_state, error=e0)
        self.current_state = initial_state
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
        # self.add_input_constraints(idx)
        self.add_terminal_constraint(idx)

        # Solve system
        try:
            sol = self.opti.solve()
        except:
            print('Failed')

        # fig, ax = plt.subplots(self.horizon)
        # for i in range(self.horizon):
        #     alphas = np.array(self.a_values[i])
        #
        #     print(alphas)
        #     ax[i].plot(np.arange(len(alphas)), alphas)
        # plt.show()

        # Update nominal state for next iteration
        self.nominal_predictions[:, 0] = sol.value(self.z0)
        self.nominal_predictions[:, 1:] = sol.value(self.Z)[:, :-1]
        self.nominal_state = sol.value(self.Z[:, 0])

        # Add tube/feedback term to action.
        action = sol.value(self.V[0]) + self.tube.K @ e0
        self.trajectory.add_action(action)
        return action

    def plot_trajectory(self, constraint=None):
        assert self.x_dim == 2, "Can only plot trajectory fro 2d systems."
        import matplotlib.pyplot as plt
        root = get_project_root()
        plt.style.use(root / 'plt_style/jz.mplstyle')
        fig, ax = plt.subplots()
        self.trajectory.plot_trajectory(ax)
        if constraint is not None:
            x = np.arange(-5, 12, 0.1)
            x = np.vstack((x, np.zeros_like(x))).T
            e = np.array([0, 0])
            y = constraint(e, x)
            ax.plot(x[:, 0], y, 'r')
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.show()
        plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', default=20, type=int,
                    help='number of scenarios')
parser.add_argument('--horizon', default=5, type=int,
                    help='num control intervals. aka horizon in dt')
parser.add_argument('--total_steps', default=10, type=int,
                    help='Total time horizon.')
parser.add_argument('--fixed_seed', dest='fix_seed', action='store_true')
parser.add_argument('--not_fixed_seed', dest='fix_seed', action='store_false')

parser.set_defaults(fix_seed=False)


if __name__ == '__main__':
    # plt.style.use('jz')  # use my special style
    np.random.seed(0)
    args = parser.parse_args()
    dim_x = 2
    dim_u = 1
    dim_noise = 2
    x0 = np.array([10, 0.0])

    # define dynamics matrices
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    Q = np.eye(dim_x)
    R = np.eye(dim_u)

    # Constraints
    state_constraint = Box(dim_x, [-1000, -2.5], [1000, 2.5])
    input_constraint = Box(dim_u, [-1000], [1000])

    # Dynamics functional
    f = lambda x, u, w: A@x + B@u + w

    # Noise
    normal_param = {'mean': np.array([0.0, -0.0]),
                    'cov': 0.01*np.eye(2)}
    normal_noise = NormalNoise(mean=normal_param['mean'],
                               cov=normal_param['cov'])
    uniform_param = {'low': -0.01, 'high': 0.01}
    uniform_noise = lambda n_samples: np.random.uniform(**uniform_param,
                                                        size=(dim_noise,
                                                              n_samples))

    lti_dyn = LTIDynamics(x0=x0,
                          A=A,
                          B=B,
                          D=np.eye(dim_noise),
                          noise=normal_noise)

    test_dyn = LTIDynamics(x0=x0,
                           A=A,
                           B=B,
                           D=np.eye(dim_noise),
                           noise=normal_noise)

    # MPC object
    # mpc = sMPC(dynamics=lti_dyn,
    #            horizon=1,
    #            n_scenarios=args.n_samples,
    #            noise=normal_noise,
    #            Q=Q,
    #            R=R,
    #            state_constraints=state_constraint,
    #            input_constraints=input_constraint,
    #            feedback='disturbance')
    # mpc = sTubeMPC(dynamics=lti_dyn,
    #                horizon=args.horizon,
    #                task_length=args.total_steps,
    #                n_scenarios=args.n_samples,
    #                noise=normal_noise,
    #                Q=Q,
    #                R=R,
    #                state_constraint=state_constraint,
    #                input_constraint=input_constraint)
    z = MX.sym('z', dim_x)
    e = MX.sym('e', dim_x)
    ellipsoid_c = Function('e_c', [z, e], [1/100 * (z[0]+e[0] - 5)**2 + 1/100 *
                                           (z[1] + e[1])**2 - 1])
    exp_c = Function('exp_c', [z, e],
                     [-5 + exp(0.1 * (z[0] + e[0])) - z[1] - e[1]])

    mpc = ConstraintTubeMPC(dynamics=lti_dyn,
                            horizon=args.horizon,
                            task_length=args.total_steps,
                            n_scenarios=args.n_samples,
                            noise=normal_noise,
                            Q=Q, R=R,
                            state_constraints=[exp_c,
                                               state_constraint.get_casadi_function()],
                            input_constraints=[input_constraint.get_casadi_function()],
                            epsilons=[0.2]*args.total_steps)
    mpc.setup_mpc(horizon=args.horizon, n_samples=args.n_samples,
                  task_length=args.total_steps)
    mpc.initialize(x0)
    normal_noise.set_mean(0.01 * np.ones(2))
    test_dyn.set_noise(normal_noise)
    traj = [test_dyn.get_state()]
    actions = []
    for i in range(args.total_steps):
        print("{}-th mpc-iteration".format(i))
        actions.append(mpc.solve_mpc_iteration(test_dyn.get_state(), i))
        traj.append(test_dyn.act(actions[-1]))
    mpc.trajectory.add(nominal=mpc.nominal_state,
                       error=test_dyn.get_state() - mpc.nominal_state)
    mpc.plot_trajectory()
