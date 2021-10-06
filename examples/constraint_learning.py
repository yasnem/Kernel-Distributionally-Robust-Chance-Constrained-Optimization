import numpy as np
from drccp_utils.rkhs_utils import mmd_eps, median_heuristic
from drccp_utils.stats_util import grid_around_data_support
from drccp_utils.utils import eval_solution
from drccp.cvx_drccp import cvx_exact_kernel, cas_const_learn
from drccp_utils.distributions import GaussianDistribution
from drccp.const_learn import (train_model, create_training_data,
                               create_decision_vars, ConstraintDataset)
import argparse
import torch
import gpytorch


# Define constraint function
def f_constraint(x, X):
    """
    Constraint function

    x: cp.Variable -- decision variable (dim,)
    X: ndarray -- Samples (n_samples, dim)
    """
    #     f = cp.exp(0.1 * (x[0, :] + X[:,0])) + x[1, :] + X[:, 1] -10

    f = -X @ x + 1

    return f


np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--n_samples', type=int, default=50)
parser.add_argument('--risk_level', type=float, default=0.1)
parser.add_argument('--big_M', type=int, default=10000)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--ml_model', type=str, default="KRR")
parser.add_argument('--output', type=str, default='f_val')

parser.add_argument('--fixed_seed', dest='fix_seed', action='store_true')
parser.add_argument('--not_fixed_seed', dest='fix_seed', action='store_false')
parser.set_defaults(fix_seed=False)


if __name__ == "__main__":
    args = parser.parse_args()
    np.random.seed(11221)

    # Define objective
    c = np.random.randint(1, 50, size=(args.dim, 1))

    def objective(x):
        return c.T @ x

    # Create uncertainty and compute parameters
    dist = GaussianDistribution(args.dim, 1.2 * np.ones(args.dim), std=0.2 * np.eye(args.dim))
    alpha = args.risk_level
    epsilon = mmd_eps(args.n_samples, alpha=alpha)
    # epsilon = 0.1
    print("MMD bound: ", epsilon)

    # MIP relaxation of problem with its solution
    samples = dist.sample(args.n_samples)
    sigma, _ = median_heuristic(samples, samples)
    cert_points = grid_around_data_support(10, samples, 0.2)
    x_sol, _, _, Eg, g_norm = cvx_exact_kernel(objective=objective,
                                               f_constraint=f_constraint,
                                               X=samples,
                                               alpha=alpha,
                                               epsilon=epsilon,
                                               M=args.big_M,
                                               sigma=sigma,
                                               supp_points=cert_points)
    y_sol = np.array([Eg, g_norm])

    # Create set of decision variable for training set
    low = 0 * np.ones(args.dim)
    high = 2 * np.ones(args.dim)
    steps = 0.1 * np.ones(args.dim)
    xx = create_decision_vars(low, high, steps)
    X, Y = create_training_data(xx=xx,
                                f_const=f_constraint,
                                samples=samples,
                                cert_pts=cert_points,
                                sigma=sigma,
                                alpha=alpha,
                                epsilon=epsilon,
                                output=args.output)

    dataset = ConstraintDataset(X, Y)
    batch_size = 20
    mode = args.ml_model
    if mode == "NN":
        model_params = {
            "in_dim": args.dim,
            "out_dim": Y.shape[1]
        }
        likelihood = None
    elif mode == "GP":
        model_params = {
            "train_x": dataset.x,
            "train_y": dataset.y,
            "in_dim": args.dim,
            "out_dim": dataset.y.shape[1],
        }
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=Y.shape[1])
    elif mode == "KRR":
        model_params = {
            "kernel": "rbf",
            "sigma": 0.25*sigma,
            "in_dim": args.dim,
            "out_dim": dataset.y.shape[1],
            "alpha": 0.001
        }
        likelihood = None
    else:
        raise ValueError

    model = train_model(dataset,
                        model_params=model_params,
                        batch_size=batch_size,
                        epochs=args.epochs,
                        lr=1e-3,
                        model_type=mode,
                        likelihood=likelihood)
    # Compute prediction error of model over the whole dataset
    Ypred = model(dataset.x)
    if type(Ypred) == torch.Tensor:
        Ypred = Ypred.detach().numpy()
    mse = np.mean(np.linalg.norm(Ypred - dataset.y.numpy(), axis=1))
    print("MSE prediction on training data: {}".format(mse))
    # visualize_model(model=model,
    #                 sol=x_sol,
    #                 y_sol=y_sol,
    #                 dim=args.dim,
    #                 X=X, Y=Y,
    #                 mode=mode,
    #                 visualization='combo',
    #                 epsilon=epsilon)

    print("Done training!")
    x_cl = cas_const_learn(objective=objective,
                           model=model,
                           samples=samples,
                           Xtrain=X,
                           cert_points=cert_points,
                           sigma=sigma,
                           alpha=alpha,
                           epsilon=epsilon,
                           constraint=f_constraint,
                           output=args.output,
                           mode=mode)

    vio_cl = eval_solution(x_cl, f_constraint, dist, 1000000)
    vio_mi = eval_solution(x_sol, f_constraint, dist, 1000000)
    print("Violation of cl soltuion: {}   Objective: {}".format(vio_cl, objective(x_cl)))
    print("Violation of mip soltuion: {}   Objective: {}".format(vio_mi, objective(x_sol)))
