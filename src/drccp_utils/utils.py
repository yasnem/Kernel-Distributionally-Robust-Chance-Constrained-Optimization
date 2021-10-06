import numpy as np
from pathlib import Path


def eval_solution(x, f_constraint, dist, n_samples):
    X_emp = dist.sample(n_samples)
    f = f_constraint(x, X_emp)
    violations = np.count_nonzero(f > 0)
    return violations / n_samples


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent