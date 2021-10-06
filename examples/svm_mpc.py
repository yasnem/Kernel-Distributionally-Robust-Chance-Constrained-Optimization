#!/usr/bin/env python
# coding: utf-8
from sklearn import svm
import torch
import dill as pickle
from torch.utils.data import Dataset
from casadi import MX, Function, vcat, exp, norm_2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH
from drcc_mpc.dr_mpc import DRCVaRTubeMPC
from drcc_mpc.dynamics import LTIDynamics, NormalNoise, UniformNoise, Trajectory
from drcc_mpc.constraint_sets import Box
from drccp_utils.rkhs_utils import median_heuristic

np.set_printoptions(suppress=True)


def make_constraint_data(center, size, n_points):
    """
    Create data to train an svm.

    Parameters
    ----------
    center: ndarray
        (dim,)
    size: ndarray
        (dim,)
    n_points: int

    Returns
    -------
    X: ndarray
        (n_points, dim)
    Y: ndarray
        (n_points, 1)
    """
    xx = np.linspace(center - size*0.7, center + size*0.7, num=int(np.sqrt(n_points)))
    x, y = np.meshgrid(xx[:, 0], xx[:, 1])
    X = np.vstack((x.ravel(), y.ravel())).T
    x_delta = 1 * np.random.uniform(-1, 1, size=X.shape)
    Y = np.logical_and(np.abs(X[:, 0] - center[0]) <= size[0] / 2,
                       np.abs(X[:, 1] - center[1]) <= size[1] / 2)
    X += x_delta
    return X, Y.reshape((-1,1))


def train_svm(c, gamma, X, Y):
    clf = svm.SVC(C=c, gamma=gamma)
    clf.fit(X, Y)
    return clf


def casadi_svm(clf):
    dim = clf.support_vectors_.shape[1]
    x = MX.sym('x', dim)
    y = MX.sym('y', dim)
    casadi_rbf = Function('rbf', [x, y],
                          [exp(-clf.gamma * norm_2(x-y)**2)])
    z = MX.sym('z', dim)
    e = MX.sym('e', dim)
    ki = vcat([casadi_rbf(z + e, s_vec) for s_vec in clf.support_vectors_])
    y_pred = clf.dual_coef_ @ ki + clf.intercept_
    cas_svm = Function('svm', [z, e], [-y_pred])
    return cas_svm


def get_cas_svm(X, Y):
    _, gamma = median_heuristic(X, X)
    clf = train_svm(10, gamma, X, Y)
    return casadi_svm(clf), clf, X, Y


class ConstraintDataset(Dataset):
    def __init__(self, decision_vars, output):
        self.x = torch.from_numpy(decision_vars).float()
        self.y = torch.from_numpy(output).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]


def plot(data, data_dir):
    # plot stuff
    mpc_traj = data['mpc_traj']
    trajs = data['trajs']
    center = data['center']
    size = data['size']
    svm_model = data['svm_model']
    plt.rcParams.update(NEURIPS_RCPARAMS)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1, figsize=(LINE_WIDTH/2, LINE_WIDTH/3))
    xx, yy = np.meshgrid(np.linspace(center[0] - size[0],
                                     center[0] + size[0], 500),
                         np.linspace(center[1] - size[1],
                                     center[1] + size[1], 500))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    Z = np.asarray(svm_model.predict(X_plot), dtype=np.float)
    Z = Z.reshape(xx.shape)
    contours = ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors=['r'],
                          linestyles='dashed')

    # mpc_traj.plot_trajectory(ax, color='k', alpha=1, label='traj', zorder=2)
    for traj in trajs:
        traj.plot_trajectory(ax, traj_only=True, marker='', zorder=1)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.set_xlim(-7, 17)
    ax.set_ylim(-5, 6.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_dir / 'drcc_mpc.pdf', dpi=300)
    plt.show()
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--risk_level', type=float, default=0.05)
parser.add_argument('--n_samples', type=int, default=40)
parser.add_argument('--n_traj', type=int, default=5)
parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--total_steps', type=int, default=10)
parser.add_argument('--compute', dest='compute', action='store_true')
parser.add_argument('--visualize', dest='compute', action='store_false')
parser.add_argument('--exp_name', type=str, default='svm_mpc')
parser.set_defaults(compute=False)


if __name__ == "__main__":
    # Create constraint
    args = parser.parse_args()
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    compute = args.compute
    if compute:
        center = np.array([5, 0])
        size = np.array([20, 6])
        n_points = 500
        X, Y = make_constraint_data(center, size, n_points)
        cas_svm, svm_model, _, _ = get_cas_svm(X, Y)
        cas_constraint = cas_svm

        dim_x = 2
        dim_u = 1
        dim_noise = 2
        x0 = np.array([10, 0.0])
        alpha = args.risk_level
        n_samples = args.n_samples
        n_traj = args.n_traj
        horizon = args.horizon
        total_steps = args.total_steps
        kernel = 'rbf'
        kernel_param = 1

        # define dynamics matrices
        A = np.array([[1, 1], [0, 1]])
        B = np.array([[0.5], [1]])
        Q = np.eye(dim_x)
        R = np.eye(dim_u)

        # Noise
        x0_dist = UniformNoise(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        normal_param = {'mean': np.array([0.0, 0.0]),
                        'cov': 0.1 * np.eye(2)}
        normal_noise = NormalNoise(mean=normal_param['mean'],
                                   cov=normal_param['cov'])
        test_noise = NormalNoise(mean=normal_param['mean'],
                                 cov=normal_param['cov'])

        # Dynamics
        lti_dyn = LTIDynamics(x0=x0,
                              A=A,
                              B=B,
                              D=np.eye(dim_noise),
                              add_noise=normal_noise,
                              multi_noise=None)

        test_dyn = LTIDynamics(x0=x0,
                               A=A,
                               B=B,
                               D=np.eye(dim_noise),
                               add_noise=test_noise,
                               multi_noise=None)
        input_constraint = Box(dim_u, [-100], [100])

        # Setup MPC
        mpc = DRCVaRTubeMPC(
            dynamics=lti_dyn,
            horizon=horizon,
            task_length=total_steps,
            n_scenarios=n_samples,
            noise=normal_noise,
            Q=Q, R=R,
            state_constraint=cas_constraint,
            input_constraint=input_constraint,
            alpha=alpha,
            kernel=kernel,
            kernel_param=1
        )
        np.random.seed(10)
        mpc.setup_mpc(horizon=horizon,
                      n_samples=n_samples,
                      alpha=alpha,
                      conf_lvl=0.99)

        # Run multiple trajectories with different initial states
        trajs = []
        for j in range(n_traj):
            np.random.seed(123 + j)
            x0_sample = x0_dist(1).squeeze() + x0
            test_dyn.reset(x0_sample)
            print(x0_sample)
            mpc.initialize(x0_sample)
            traj = Trajectory()
            traj.add(state=x0_sample)
            actions = []
            for i in range(total_steps):
                print("{}-th mpc-iteration".format(i))
                traj.add_action(mpc.solve_mpc_iteration(test_dyn.get_state(), i))
                test_dyn.act(traj.actions[-1])
                traj.add(state=test_dyn.get_state())
            mpc.trajectory.add(state=test_dyn.get_state(), nominal=mpc.nominal_state)
            trajs.append(traj)

        # Run MPC with original initial state
        print(x0)
        np.random.seed(12)
        mpc.initialize(x0)
        test_dyn.reset(x0)
        actions = []
        for i in range(total_steps):
            print("{}-th mpc-iteration".format(i))
            actions.append(mpc.solve_mpc_iteration(test_dyn.get_state(), i))
            test_dyn.act(actions[-1])
        mpc.trajectory.add(state=test_dyn.get_state(), nominal=mpc.nominal_state)
        mpc_traj = mpc.trajectory
        traj_dict = {
            'mpc_traj': mpc_traj,
            'trajs': trajs,
            'center': center,
            'size': size,
            'svm_model': svm_model
        }
        with open(data_dir / 'svm_mpc_data.pk', 'wb') as file:
            pickle.dump(traj_dict, file)
    with open(data_dir / 'svm_mpc_data.pk', 'rb') as file:
        traj_dict = pickle.load(file)
    plot(traj_dict, data_dir)
