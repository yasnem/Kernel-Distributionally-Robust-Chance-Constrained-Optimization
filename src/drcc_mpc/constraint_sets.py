"""
Set definitions used to represent probabilistic reachable sets.
"""
from casadi import (SX, MX, vertcat, norm_1, Function, nlpsol, mtimes,
                    gradient, jacobian, fmax)
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import os


class DifferentiablePRS(object):
    """Generic class to define constraints from sets of casadi.Functions."""
    def __init__(self, func=None, set=None):
        pass

    def from_set(self, set):
        pass

    def from_function(self, func):
        pass

    def contains(self, Q):
        pass


class Box(object):
    def __init__(self, n_dim, lb, ub):
        """
        Create Box set.

        Parameters
        ----------
        n_dim: int
        lb: iteratable
            Bounds must be axis ordered and lb have length n_dim.
        ub: iteratable
            Bounds must be axis ordered and ub have length n_dim.
        """
        self.n = n_dim
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        assert lb.shape[0] == n_dim, "Wrong shape for lower bounds"
        assert ub.shape[0] == n_dim, "Wrong shape for upper bounds"
        assert np.all(ub >= lb), "Upper bounds {0} must be larger than lower " \
                                "bounds {1}.".format(ub, lb)
        self.lb = lb
        self.ub = ub
        self.range = ub - lb
        A = []
        A.extend(np.eye(n_dim))
        A.extend(-np.eye(n_dim))
        self.A = np.array(A, ndmin=2)
        self._b = self._get_b_from_bounds(lb, ub)

    def _get_b_from_bounds(self, lb, ub):
        return np.hstack((ub, -lb)).reshape((2*self.n, 1))

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, b):
        return self._set_b(b)

    def set_bounds(self, lb, ub):
        lb = np.asarray(lb)
        ub = np.asarray(ub)
        assert lb.shape[0] == self.n, "Wrong shape for lower bounds"
        assert ub.shape[0] == self.n, "Wrong shape for upper bounds"
        assert np.all(ub >= lb), "Upper bounds {0} must be larger than lower " \
                                 "bounds {1}.".format(ub, lb)
        self.lb = lb
        self.ub = ub
        self.range = ub - lb
        self._b = self._get_b_from_bounds(lb, ub)

    def _set_b(self, b):
        self._b = np.squeeze(b).reshape((2*self.n, 1))
        self.ub = self._b[:self.n, 0]
        self.lb = -self._b[self.n:, 0]
        self.range = self.ub - self.lb

    def plot(self, ax, center=None, **kwargs):
        """
        Add box to axes object.

        Parameters
        ----------
        ax: matplotlib.Axes
        """
        if self.n != 2:
            raise NotImplementedError("Only have implementations for rectangles.")
        xy_anchor = self.lb
        if center is not None:
            assert np.all(center.shape == xy_anchor.shape), "Wrong shape of desired center."
            xy_anchor = center + self.lb
        poly = Rectangle(xy_anchor, width=self.range[0], height=self.range[1],
                         **kwargs)
        ax.add_patch(poly)

    def contains(self, Q):
        """
        Check if set of points Q is contained in the box.

        Parameters
        ----------
        Q: ndarray
            Set of points with shape (self.n, num_points)

        Returns
        -------
        boolean array of shape (num_points,)
        """
        # TODO(yassine): Replace this with np.reshape(Q, (self.n, -1))
        x = np.atleast_2d(np.squeeze(Q).astype(float))  # x: one or more points
        if x.shape[
            0] == 1:  # make x a column if it is single point (here a row)
            x = x.T
        elif x.shape[0] != self.n:
            raise ValueError(
                f'Testing if a box in R^{self.n} contains points'
                f'in R^{x.shape[0]} is not allowed: points must be '
                f'columns (transpose the input?)')
        # Check if every point satisfies every inequality.
        Axb = self.A @ x - self.b
        x_contained = np.logical_or(Axb < 0, np.isclose(Axb, 0))
        return np.all(x_contained, axis=0)

    def reset(self):
        """Reset box."""
        self.set_bounds(np.zeros(self.n), np.ones(self.n))

    def union(self, box):
        # TODO(yassine): Think how to implement if sets don't intersect
        pass

    def intersection(self, box):
        """
        Compute intersection of two box sets.

        Parameters
        ----------
        box: Box

        Returns
        -------

        """
        new_lb = np.maximum(self.lb, box.lb)
        new_ub = np.minimum(self.ub, box.ub)
        self.set_bounds(new_lb, new_ub)

    def pontryagin_difference(self, box):
        """
        Compute pontryagin difference wrt argument.

        P - Q = {x in P | x + q in P, for all q in Q}
        If Q contains the origin, it will act like a layer removed from P,
        resulting in the solution.

        Parameters
        ----------
        box: Box

        Returns
        -------
        Box
        """
        # Dimension must match.
        assert self.n == box.n
        new_ub = self.ub - box.ub
        new_lb = self.lb - box.lb
        if np.any(new_lb >= new_ub):
            print('Test')
            return Box(self.n, np.zeros_like(new_lb), np.zeros_like(new_ub))
        else:
            return Box(self.n, new_lb, new_ub)

    def save(self, save_dir, name):
        np.save(os.path.join(save_dir, name), [self.lb, self.ub])

    @classmethod
    def load(cls, save_dir, name):
        bounds = np.load(os.path.join(save_dir, name))
        return cls(n_dim=bounds.shape[1], lb=bounds[:, 0], ub=bounds[:, 1])

    @classmethod
    def from_bounds(cls, bounds):
        return cls(n_dim=bounds.shape[1], lb=bounds[:, 0], ub=bounds[:, 1])

    def to_bounds(self):
        return np.vstack((self.lb, self.ub))

    def get_casadi_function(self):
        z = MX.sym('z', self.n)
        e = MX.sym('e', self.n)
        c_func = Function('c', [z, e], [self.A @ z + self.A @ e - self.b])
        return c_func


def fit_box_to_scenarios(scenarios, box):
    """
    Fit Box PRS to scenarios by solving optimization problem.

    Parameters
    ----------
    scenarios: ndarray
        Scenarios of shape (box.n, n_scenarios)
    box: Box
        Probabilistic Reachable Set
    """
    # ranges = np.zeros((box.n, 2))
    # ranges[:, 0] = np.min(scenarios, axis=1)
    # ranges[:, 1] = np.max(scenarios, axis=1)
    # for i in range(box.n):
    #     if np.all(ranges[i, :] < 0):
    #         ranges[i, 1] = 0
    #     elif np.all(ranges[i, :] > 0):
    #         ranges[i, 0] = 0
    #     else:
    #         continue
    #
    # box.set_bounds(ranges[:, 0], ranges[:, 1])
    b = SX.sym('b', 2 * box.n)
    A = SX.sym('A', box.A.shape[0], box.A.shape[1])
    A[:, :] = box.A
    assert scenarios.shape[0] == box.A.shape[1]
    Axb = []
    for i in range(scenarios.shape[1]):
        x_s = SX.sym('x_s' + str(i), scenarios.shape[0])
        x_s[:] = scenarios[:, i]
        Axb.append(mtimes(A, x_s) - b)
    args = {'f': norm_1(b), 'x': b, 'g': vertcat(*Axb)}
    solver = nlpsol('solver', 'ipopt', args, {'ipopt.print_level': 0,
                                              'print_time': False})
    sol = solver(ubg=0)
    b = np.array(sol['x'])
    box.b = b


if __name__ == "__main__":
    scenarios = np.random.uniform(-1, 1, size=(2, 5))
    fig, ax = plt.subplots()
    ax.scatter(scenarios[0, :], scenarios[1, :], facecolor='black')
    box = Box(2, [-2, -2], [2, 2])
    box.plot(ax, facecolor='blue', edgecolor='black', alpha=0.2)
    prs = Box(2, [-1, -1], [1, 1])
    # Fit box to scenarios
    fit_box_to_scenarios(scenarios, prs)
    prs.plot(ax, facecolor='red', edgecolor='black', alpha=0.7)

    diff = box.pontryagin_difference(prs)
    diff.plot(ax, facecolor='green', edgecolor='black', alpha=0.6)
    ax.autoscale(enable=True)
    plt.show()
