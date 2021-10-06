from abc import ABCMeta, abstractmethod, abstractclassmethod
import os
import argparse
import numpy as np
from casadi import mtimes, Function, MX, mmax, vec
import matplotlib.pyplot as plt
from drcc_mpc.constraint_sets import Box, fit_box_to_scenarios
from drcc_mpc.lqr import dt_lqr
from drcc_mpc.dynamics import load_noise


class BaseTube(metaclass=ABCMeta):
    def __init__(self, A, B, Q, R, task_len, n_samples, noise):
        """
        Parameters
        ----------
        A: ndarray
            System Matrix A
        B: ndarray
            Input Matrix B
        Q: ndarray
            Quadratic cost matrix for state
        R: ndarray
            Quadratic cost matrix for input
        horizon: int
        n_samples: int
        noise: callable
        """
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

        self.dim_x = A.shape[1]
        self.dim_u = B.shape[1]
        # Only consider additive noise for now.
        self.dim_noise = A.shape[0]

        self.K, _ = dt_lqr(A, B, Q, R)
        self.e = MX.sym('e', A.shape[0])
        self.w = MX.sym('w', A.shape[0])
        self.A_err = (self.A + self.B @ self.K)
        self.err_dyn = Function('f_err', [self.e, self.w],
                                [mtimes(self.A_err, self.e) + self.w])

        self.task_len = task_len
        self.n_samples = n_samples
        self.noise = noise
        self.scenarios = np.zeros((self.task_len + 1,
                                   self.n_samples,
                                   self.dim_x))
        self.fitted = False

    def _generate_scenarios(self, e_0):
        """Generate scenarios given e_0 for assumed noise distribution."""
        self.scenarios[0, :, :] = e_0
        for i in range(self.task_len):
            disturbance_samples = self.noise(self.n_samples)
            for j in range(self.n_samples):
                self.scenarios[i + 1, j, :] = self.A_err @ self.scenarios[i, j, :]
                self.scenarios[i + 1, j, :] += disturbance_samples[:, j]

    def setup(self, task_len, n_samples):
        self.task_len = task_len
        self.n_samples = n_samples
        self.scenarios = np.zeros((self.task_len + 1,
                                   self.n_samples,
                                   self.dim_x))
        self.fitted = False

    @classmethod
    def from_dict(cls, tube_config):
        noise = load_noise(tube_config)
        return cls(A=np.array(tube_config['A']),
                   B=np.array(tube_config['B']),
                   Q=np.array(tube_config['Q']),
                   R=np.array(tube_config['R']),
                   task_len=tube_config['total_steps'],
                   n_samples=tube_config['n_samples'],
                   noise=noise)

    def fit_tube(self, e0):
        self._generate_scenarios(e0)
        self.fitted = True

    def reset_tube(self):
        self.scenarios = np.zeros((self.task_len + 1,
                                   self.n_samples,
                                   self.dim_x))
        self.fitted = False

    def save_scenarios(self, dir, name):
        assert self.fitted
        name += '.npy'
        with open(os.path.join(dir, name), 'wb') as f:
            np.save(f, self.scenarios)


class PRSTube(BaseTube):
    """
    Scenario generation and PRS fitting for LTI system and Quadratic cost.
    """
    def __init__(self, A, B, Q, R, task_len, n_samples, noise):
        """
        Parameters
        ----------
        A: ndarray
            System Matrix A
        B: ndarray
            Input Matrix B
        Q: ndarray
            Quadratic cost matrix for state
        R: ndarray
            Quadratic cost matrix for input
        horizon: int
        n_samples: int
        noise: callable
        """
        super(PRSTube, self).__init__(A, B, Q, R, task_len, n_samples, noise)
        self.state_prs = [Box(self.dim_x,
                              np.zeros(self.dim_x),
                              np.ones(self.dim_x)) for j in range(task_len)]
        self.input_prs = [Box(self.dim_u,
                              np.zeros(self.dim_u),
                              np.ones(self.dim_u)) for j in range(task_len)]

    def setup(self, task_len, n_samples):
        self.task_len = task_len
        self.n_samples = n_samples
        self.state_prs = [Box(self.dim_x,
                              np.zeros(self.dim_x),
                              np.ones(self.dim_x)) for j in range(task_len)]
        self.input_prs = [Box(self.dim_u,
                              np.zeros(self.dim_u),
                              np.ones(self.dim_u)) for j in range(task_len)]
        self.scenarios = np.zeros((self.task_len + 1,
                                   self.n_samples,
                                   self.dim_x))


    def _fit_state_constraints(self):
        """Fit probabilistic reachabe sets to get state restrictions."""
        # TODO(yassine): Make sure scenarios were already generated.
        for i in range(self.task_len):
            fit_box_to_scenarios(self.scenarios[i+1, :, :].T, self.state_prs[i])

    def _fit_input_constraints(self):
        """Fit PRS to get input restrictions."""
        for i in range(self.task_len):
            fit_box_to_scenarios(self.K @ self.scenarios[i+1, :, :].T,
                                 self.input_prs[i])

    def fit_tube(self, e0):
        self._generate_scenarios(e0)
        self._fit_state_constraints()
        self._fit_input_constraints()
        self.fitted = True

    def reset_tube(self):
        self.scenarios = np.zeros((self.task_len + 1,
                                   self.n_samples,
                                   self.dim_x))
        for prs in self.state_prs:
            prs.reset()
        for prs in self.input_prs:
            prs.reset()
        self.fitted = False

    def plot_scenarios_boxs(self, n_rows):
        """Plot scenarios per timestep and fitted sets."""
        fig, ax_arr = plt.subplots(n_rows,
                                   int(np.ceil((self.task_len / n_rows))))
        k = int(np.ceil(self.task_len / n_rows))
        for i, prs in enumerate(self.state_prs):
            prs.plot(ax_arr[int(i / k), i % k], alpha=0.5)
            ax_arr[int(i / k), i % k].scatter(x=self.scenarios[i, :, 0],
                                              y=self.scenarios[i, :, 1],
                                              marker='x',
                                              facecolor='black')
            ax_arr[int(i / k), i % k].set_xlim(-1, 1)
            ax_arr[int(i / k), i % k].set_ylim(-1, 1)


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', default=20, type=int,
                    help='number of scenarios')
parser.add_argument('--horizon', default=20, type=int,
                    help='num control intervals. aka horizon in dt')


if __name__ == '__main__':
    args = parser.parse_args()
    # define dynamics
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])

    Q = np.eye(A.shape[0])
    R = np.eye(B.shape[1])

    dim_noise = 2
    normal_param = {'mean': np.array([0, 0]), 'cov': 0.01*np.eye(2)}
    normal_noise = lambda n_samples: np.random.multivariate_normal(
        **normal_param, size=(n_samples,)).T

    uniform_param = {'low': -0.01, 'high': 0.01}
    uniform_noise = lambda n_samples: np.random.uniform(**uniform_param,
                                                        size=(dim_noise,
                                                              n_samples))

    tube = BaseTube(A, B, Q, R, horizon=args.horizon, n_samples=args.n_samples,
                     noise=normal_noise)
    tube.fit_tube(np.array([0, 0]))
    tube.plot_scenarios_boxs(4)
    X_box = Box(A.shape[0], [-1, -1], [1, 1])

    k = int(np.ceil(args.horizon / 4))
    fig, ax_arr = plt.subplots(4, int(np.ceil((args.horizon / 4))))
    for i, prs in enumerate(tube.state_prs):
        tight_box = X_box.pontryagin_difference(prs)
        tight_box.plot(ax_arr[int(i / k), i % k], alpha=0.5)
        X_box.plot(ax_arr[int(i / k), i % k], alpha=0.2)
        ax_arr[int(i / k), i % k].autoscale(enable=True)
    plt.show()
