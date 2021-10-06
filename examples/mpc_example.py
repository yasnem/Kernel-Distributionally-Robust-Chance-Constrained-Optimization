import argparse
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import pickle
from pathlib import Path

from drccp_utils.rkhs_utils import mmd_eps, median_heuristic
from drcc_mpc.dynamics import LTIDynamics, NormalNoise, UniformNoise, Trajectory
from drcc_mpc.constraint_sets import Box
from drcc_mpc.constraints import DifferentiableConstraint
from drcc_mpc.dr_mpc import exponential_constraint, DRCVaRTubeMPC
from drcc_mpc.st_mpc import sMPC, ConstraintTubeMPC


parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', default=20, type=int,
                    help='number of scenarios')
parser.add_argument('--horizon', default=10, type=int,
                    help='num control intervals. aka horizon in dt')
parser.add_argument('--total_steps', default=10, type=int,
                    help='Total time horizon.')
parser.add_argument('--risk_level', type=float, default=0.1)
parser.add_argument('--mpc_type', type=str, default="drcc")
parser.add_argument('--kernel', type=str, default='rbf')
parser.add_argument('--fixed_seed', dest='fix_seed', action='store_true')
parser.add_argument('--not_fixed_seed', dest='fix_seed', action='store_false')

parser.set_defaults(fix_seed=False)


if __name__ == '__main__':
    # plt.style.use('jz')  # use my special style
    # np.random.seed(0)
    args = parser.parse_args()
    dim_x = 2
    dim_u = 1
    dim_noise = 2
    x0 = np.array([10, 0.0])
    alpha = args.risk_level

    # define dynamics matrices
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0.5], [1]])
    Q = np.eye(dim_x)
    R = np.eye(dim_u)

    # Noise
    normal_param = {'mean': np.array([0.0, 0.0]),
                    'cov': 0.1 * np.eye(2)}
    normal_noise = NormalNoise(mean=normal_param['mean'],
                               cov=normal_param['cov'])
    test_noise = NormalNoise(mean=normal_param['mean'],
                             cov=normal_param['cov'])
    test_noise.set_mean(-0.5 * np.ones_like(normal_param['mean']))
    test_noise.set_cov(0.1 * np.ones(normal_param['cov'].shape))

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

    epsilon = mmd_eps(args.n_samples, alpha=alpha)
    # epsilon = 0.1
    x_lvl = -4
    growth_rate = 0.1
    cas_constraint = exponential_constraint(x_lvl, growth_rate)
    def constraint(z, e):
        return -1 + x_lvl + np.exp(growth_rate * (z[0] + e[:, 0])) - z[1] - e[:, 1]
    state_constraint = Box(dim_x, [-100, -2.5], [100, 2.5])
    input_constraint = Box(dim_u, [-100], [100])
    if args.mpc_type == 'scenario':
        mpc = sMPC(
            dynamics=lti_dyn,
            horizon=args.horizon,
            n_scenarios=args.n_samples,
            noise=normal_noise,
            Q=Q, R=R,
            state_constraint=state_constraint,
            input_constraint=input_constraint,
            feedback='disturbance'
        )
    elif args.mpc_type == 'drcc':
        mpc = DRCVaRTubeMPC(
            dynamics=lti_dyn,
            horizon=args.horizon,
            task_length=args.total_steps,
            n_scenarios=args.n_samples,
            noise=normal_noise,
            Q=Q, R=R,
            state_constraint=cas_constraint,
            input_constraint=input_constraint,
            alpha=alpha,
            epsilon=epsilon,
            kernel=args.kernel,
            kernel_param=1
        )
    elif args.mpc_type == 'ctube':
        mpc = ConstraintTubeMPC(
            dynamics=lti_dyn,
            horizon=args.horizon,
            task_length=args.total_steps,
            n_scenarios=args.n_samples,
            noise=normal_noise,
            Q=Q, R=R,
            state_constraints=[state_constraint.get_casadi_function()],
            input_constraints=[input_constraint.get_casadi_function()],
            epsilons=[epsilon]*args.total_steps,
            method='gradient'
        )
    else:
        raise ValueError
    mpc.setup_mpc(horizon=args.horizon,
                  n_samples=args.n_samples,
                  task_length=args.total_steps,
                  alpha=alpha,
                  epsilon=epsilon)
    mpc.initialize(x0)

    traj = [test_dyn.get_state()]
    actions = []
    for i in range(args.total_steps):
        print("{}-th mpc-iteration".format(i))
        actions.append(mpc.solve_mpc_iteration(test_dyn.get_state(), i))
        traj.append(test_dyn.act(actions[-1]))
    mpc.trajectory.add(state=test_dyn.get_state(), nominal=mpc.nominal_state)
    mpc.plot_trajectory(constraint)
