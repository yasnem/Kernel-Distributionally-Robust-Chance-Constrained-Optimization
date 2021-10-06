import numpy as np
from scipy.stats import norm
from drccp_utils.distributions import GaussianDistribution, GaussianMixtureDistribution
from drccp_utils.plot_utils import NEURIPS_RCPARAMS, LINE_WIDTH

import argparse
from pathlib import Path

np.set_printoptions(suppress=True)

def create_figure(samples, dist, fig_dir, stem=True, n_samples=1000):
    """Create Majorization figure of optimized RKHS-function."""
    if n_samples > 0:
        stat_samples = dist.sample(n_samples).squeeze()
        stat_samples.sort()
        alpha = 0.3
        var = stat_samples[int((1-alpha)*n_samples)]
        cvar = np.mean(stat_samples[int((1-alpha) * n_samples):])
    import matplotlib.pyplot as plt
    plt.rcParams.update(NEURIPS_RCPARAMS)
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1, figsize=(7*LINE_WIDTH/12, LINE_WIDTH/3))
    if stem:
        markerline, stemlines, baseline = ax.stem(samples, 0.05 * np.ones_like(samples),
                                                  linefmt='grey', markerfmt='rD')
    x_axis = np.arange(-2, 7, 0.01)
    pdf = norm.pdf(x_axis, dist.means[0], dist.stds[0]) * dist.weights[0] + norm.pdf(x_axis, dist.means[1], dist.stds[1]) * dist.weights[1]
    ax.plot(x_axis,
            pdf.squeeze())
    ax.hist(samples, density=True)
    if n_samples > 0:
        ax.axvline(var, ymin=0, ymax=0.5, linestyle='--', c='r', lw=0.8)
        ax.text(var+0.1, .3, 'VaR')
        ax.axvline(cvar, ymin=0, ymax=0.37, linestyle='--', c='r', lw=0.8)
        ax.text(cvar+0.1, .2, 'CVaR')
        ax.arrow(var, 0.4, 7-var, 0, shape='full', ls='-', lw=0.8, head_width=0.03)
        ax.text(4, 0.42, r'$\alpha$')
    ax.set_xlabel(r'$\xi$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'density_stem.pdf', dpi=300)
    plt.show()
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=1)
parser.add_argument('--n_samples', type=int, default=15)
parser.add_argument('--stem', dest='stem', action='store_true')
parser.add_argument('--no_stem', dest='stem', action='store_false')
parser.set_defaults(stem=True)


if __name__ == "__main__":
    args = parser.parse_args()
    n_samples = args.n_samples
    dim = args.dim
    m1 = np.array([0.3])
    m2 = np.array([2.7])
    s1 = 0.4 * np.eye(1)
    s2 = 1.5 * np.eye(1)
    dist = GaussianMixtureDistribution(dim, means=[m1, m2], stds=[s1, s2], weights=[0.7, 0.3])
    samples = dist.sample(n_samples)
    samples.sort()
    alpha = 0.1
    var = samples[int(1-alpha)*len(samples)]
    cvar = np.mean(samples[int(1-alpha)*len(samples):])
    fig_dir = Path(__file__).parent / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    create_figure(samples, dist, fig_dir, stem=args.stem, n_samples=0)