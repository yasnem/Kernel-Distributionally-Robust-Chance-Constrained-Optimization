import numpy as np
from drccp_utils.rkhs_utils import compute_bootstrap_rkhs_radius
from drccp_utils.distributions import GaussianDistribution

import argparse
from pathlib import Path

np.set_printoptions(suppress=True)


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=2)
parser.add_argument('--risk_level', type=float, default=0.05)
parser.add_argument('--kernel', type=str, default='rbf')


if __name__ == "__main__":
    args = parser.parse_args()
    mean = np.zeros(args.dim)
    std = 75 * np.eye(args.dim)
    dist = GaussianDistribution(args.dim, mean, std=std)
    # dist = GaussianDistribution(args.dim, 0 * np.ones(args.dim), std=1 * np.eye(args.dim))
    alpha = args.risk_level
    n_samples = [20, 30, 40, 50]
    fig_dir = Path(__file__).parent / 'data'
    fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for n_sample in n_samples:
        np.random.seed(13)
        X = dist.sample(n_sample)
        eps = compute_bootstrap_rkhs_radius(X, plot=True, fig_dir=fig_dir)
