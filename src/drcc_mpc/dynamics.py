import numpy as np
import os
from abc import ABCMeta, abstractmethod, abstractclassmethod


class BaseDynamics(metaclass=ABCMeta):

    def __init__(self, x0, f, add_noise, multi_noise):
        self.x0 = x0
        self.x = x0
        self.add_noise = add_noise
        self.multi_noise = multi_noise
        self.dyn = f
        super().__init__()

    @abstractmethod
    def act(self, u):
        pass

    def get_state(self):
        return self.x

    def reset(self):
        self.x = np.copy(self.x0)


class GeneralDynamics(BaseDynamics):
    def __init__(self, x0, f, noise, noise_dim):
        """Set up dynamics with initial state and noise dist."""
        super(GeneralDynamics, self).__init__(x0=x0, f=f, add_noise=noise,
                                              multi_noise=None)
        self.noise_dim = noise_dim

    def act(self, u):
        w = self.add_noise(1)
        x_new = self.dyn(self.x, u, w)
        self.x = np.asarray(x_new)[:, 0]
        return self.x


class LTIDynamics(BaseDynamics):
    def __init__(self, x0, A, B, D, add_noise, multi_noise):
        f = lambda x, u, w: A@x + B@u + D@w
        super(LTIDynamics, self).__init__(x0=x0, f=f, add_noise=add_noise,
                                          multi_noise=multi_noise)
        self.A = A
        self.B = B
        self.D = D
        self.noise_dim = D.shape[1]

    def act(self, u):
        w = self.add_noise(1)[:, 0]
        if self.multi_noise is None:
            delta = 0
        else:
            delta = self.multi_noise(1)
        self.x = (self.A + delta)@self.x + self.B@u + self.D@w
        return self.x

    def add_shift(self, shift):
        self.A += shift
        f = lambda x, u, w: self.A @ x + self.B @ u + self.D @ w
        super(LTIDynamics, self).__init__(x0=self.x0, f=f,
                                          add_noise=self.add_noise,
                                          multi_noise=self.multi_noise)

    def reset(self, x0=None):
        if x0 is not None:
            self.x0 = x0
        self.x = self.x0

    @classmethod
    def from_config(cls, config):
        if config['add_noise']['type'] == 'gaussian':
            noise = NormalNoise.from_dict(config)
        elif config['add_noise']['type'] == 'uniform':
            noise = UniformNoise.from_dict(config)
        else:
            raise ValueError('Invalid noise!')
        return cls(x0=np.array(config['x0']),
                   A=np.array(config['A']),
                   B=np.array(config['B']),
                   D=np.array(config['D']),
                   add_noise=noise,
                   multi_noise=None)


def load_noise(noise_config):
    """Load noise object from dict config."""
    if noise_config['type'] == 'gaussian':
        mean = np.array(noise_config['mean'])
        cov = np.array(noise_config['cov'])
        noise = NormalNoise(mean, cov)
    elif noise_config['type'] == 'uniform':
        lb = np.array(noise_config['lb'])
        ub = np.array(noise_config['ub'])
        noise = UniformNoise(low=lb, high=ub)
    else:
        raise ValueError('Invalid noise type.')
    return noise


class BaseNoise(metaclass=ABCMeta):
    def __call__(self, n_samples):
        pass

    def to_dict(self):
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config):
        pass


class NormalNoise(BaseNoise):
    def __init__(self, mean, cov):
        """
        Wrapper for np.random.multivariate_normal.

        Parameters
        ----------
        mean: ndarray
            A (n,) mean vector of the distribution.
        cov: ndarray
            A (n, n) covariance matrix of the distribution.
        """
        self.dim = len(mean)
        self.mean = mean
        self.cov = cov

    def __call__(self, n_samples):
        return np.random.multivariate_normal(mean=self.mean,
                                             cov=self.cov,
                                             size=(n_samples,)).T

    def to_dict(self):
        return {'type': 'gaussian',
                'mean': self.mean.tolist(),
                'cov': self.cov.tolist()}

    @classmethod
    def from_dict(cls, config):
        return cls(mean=np.array(config['mean']),
                   cov=np.array(config['cov']))

    def set_mean(self, mean):
        self.mean = mean

    def set_cov(self, cov):
        self.cov = cov


class MultiplicativeNormalNoise(BaseNoise):
    def __init__(self, mean, cov, shape):
        """
        Wrapper for np.random.multivariate_normal.

        Parameters
        ----------
        mean: float
            Mean of the distribution.
        cov: float
            Covariance matrix of the distribution.
        """
        self.shape = shape
        self.mean = mean
        self.cov = cov

    def __call__(self, n_samples):
        return np.random.normal(loc=self.mean,
                                scale=self.cov,
                                size=self.shape)

    def to_dict(self):
        return {'type': 'gaussian',
                'mean': self.mean.tolist(),
                'cov': self.cov.tolist()}

    @classmethod
    def from_dict(cls, config):
        return cls(mean=float(config['mean']),
                   cov=float(config['cov']),
                   shape=np.array(config['shape']))

    def set_mean(self, mean):
        self.mean = mean

    def set_cov(self, cov):
        self.cov = cov


class UniformNoise(BaseNoise):
    def __init__(self, low, high):
        """
        Wrapper for np.random.uniform.

        Parameters
        ----------
        low: ndarray
            A (n,) lower bound vector
        high: ndarray
            A (n,) upper bound vector
        """
        self.low = low
        self.high = high
        self.dim = len(low)

    def __call__(self, n_samples):
        return np.random.uniform(low=self.low,
                                 high=self.high,
                                 size=(n_samples,
                                       self.dim)).T

    def to_dict(self):
        return {'type': 'uniform',
                'low': self.low.tolist(),
                'high': self.high.tolist()}

    @classmethod
    def from_dict(cls, config):
        return cls(low=np.array(config['low']),
                   high=np.array(config['high']))


class Trajectory(object):
    def __init__(self):
        self.nominal_trajectory = []
        self.error_trajectory = []
        self.state_trajectory = []
        self.actions = []

    def get_len(self):
        return len(self.state_trajectory)

    def eval_performance(self, cost_func, type):
        # TODO(yassine): Evaluate cost for complete trajectory
        pass

    def eval_stateconstraint(self, constraint_lst):
        """
        Evaluate state constraint function.

        Parameters
        ----------
        constraint_lst: [casadi.Function, ..]
            A function c(z, e) taking nominal and error state as arguments.

        Returns
        -------
        c_arr: ndarray
            An array with constraint function evaluation.
        """
        traj_len = len(self.state_trajectory)
        c_lst = []
        for constraint in constraint_lst:
            c_arr = np.zeros((constraint.size_out(0)[0], traj_len))
            for i in range(traj_len):
                c_arr[:, i] = constraint(self.nominal_trajectory[i],
                                         self.error_trajectory[i]).toarray().squeeze()
            c_lst.append(c_arr)
        return np.vstack(c_lst)

    def eval_stateviolation(self, constraint_lst):
        """
        Evaluate violation of constraint function, i.e. c(z, e) > 0.

        Parameters
        ----------
        constraint: casadi.Function
            A function c(z, e) taking nominal and error state as arguments.

        Returns
        -------
        vio_arr: ndarray
            An array with boolean.
        """
        traj_len = len(self.state_trajectory)
        vio_lst = []
        for constraint in constraint_lst:
            vio_arr = np.zeros(traj_len, dtype=bool)
            for i in range(traj_len):
                vio_arr[i] = np.all(constraint(self.nominal_trajectory[i],
                                               self.error_trajectory[i]) <= 0)
            vio_lst.append(vio_arr)
        return np.vstack(vio_lst)

    def add(self, **kwargs):
        if 'state' in kwargs and 'nominal' in kwargs:
            self.state_trajectory.append(kwargs['state'])
            self.nominal_trajectory.append(kwargs['nominal'])
            self.error_trajectory.append(kwargs['state'] - kwargs['nominal'])
        elif 'nominal' in kwargs and 'error' in kwargs:
            self.nominal_trajectory.append(kwargs['nominal'])
            self.error_trajectory.append(kwargs['error'])
            self.state_trajectory.append(kwargs['nominal'] + kwargs['error'])
        elif 'state' in kwargs:
            self.state_trajectory.append(kwargs['state'])
            self.nominal_trajectory.append(kwargs['state'])
            self.error_trajectory.append(np.zeros_like(kwargs['state']))
        else:
            raise ValueError("You did not pass valid arguments.")

    def add_action(self, action):
        self.actions.append(action)

    def reset(self):
        self.state_trajectory = []
        self.nominal_trajectory = []
        self.error_trajectory = []

    def plot_trajectory(self, ax, traj_only=False, **kwargs):
        assert len(self.state_trajectory) > 0
        assert len(self.state_trajectory[0]) == 2, "Can only plot 2d trajectories."
        nominal_traj = np.array(self.nominal_trajectory)
        err_traj = np.array(self.error_trajectory)
        traj = nominal_traj + err_traj
        marker = kwargs.get('marker', 'o')
        markersize = kwargs.get('markersize', 12)
        alpha = kwargs.get('alpha', 0.6)
        color = kwargs.get('color', 'grey')
        label = kwargs.get('label', None)
        zorder = kwargs.get('zorder', 2)
        if not traj_only:
            ax.plot(nominal_traj[:, 0], nominal_traj[:, 1], marker=marker, c='orange',
                    label='nominal {}'.format(label), markersize=markersize, zorder=zorder)
        ax.plot(traj[:, 0], traj[:, 1], marker=marker, c=color,
                label=label, markersize=markersize, alpha=alpha, zorder=zorder)
        # ax.set_xlabel('$x_1$', fontsize=24)
        # ax.set_ylabel('$x_2$', fontsize=24)
        # ax.autoscale(enable=True)
        # ax.legend()

    def save_trajectory(self, dir, name):
        filename = os.path.join(dir, name + '.npz')
        with open(filename, 'wb') as f:
            np.savez(f,
                     np.array(self.nominal_trajectory),
                     np.array(self.error_trajectory),
                     np.array(self.actions))

    @classmethod
    def load_trajectory(cls, dir, name):
        traj = cls()
        npzfile = np.load(os.path.join(dir, name))
        traj.nominal_trajectory = list(npzfile['arr_0'])
        traj.error_trajectory = list(npzfile['arr_1'])
        traj.state_trajectory = list(npzfile['arr_0'] + npzfile['arr_1'])
        traj.actions = list(npzfile['arr_2'])
        return traj
