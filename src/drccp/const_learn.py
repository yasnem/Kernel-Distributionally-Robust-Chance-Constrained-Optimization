import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.kernel_ridge import KernelRidge
import gpytorch
from scipy.spatial import ConvexHull
import casadi as cas
import numpy as np
import cvxpy as cp
from multiprocessing import Pool, cpu_count

from drccp_utils.rkhs_utils import median_heuristic, rbf_kernel, rkhs_func, rkhs_func_exp, rkhs_norm
from drccp_utils.casadi_utils import NNCasadi, GPCasadi, KRRCasadi



def create_decision_vars(low, high, step):
    """Create a grid of points."""
    assert len(low) == len(high) == len(step)

    dim = len(low)
    steps = []
    for i in range(dim):
        steps.append(np.arange(low[i], high[i], step[i]))
    indices = np.meshgrid(*steps)
    decision_vars = np.vstack([idx.flatten() for idx in indices]).T
    return decision_vars


def create_training_data(xx, f_const, samples, cert_pts, sigma, alpha, epsilon, output):
    """
    Parallelize dataset creation using multiprocessing.

    Parameters
    ----------
    xx: ndarray
    f_const: callable
    samples: ndarray
    cert_pts: ndarray
    sigma: float
    alpha: float
    epsilon: float
    output: string

    Returns
    -------
    X: ndarray
    Y: ndarray
    """
    pool = Pool(cpu_count())
    res = []
    for x in xx:
        res.append(pool.apply_async(cvx_training_data, (f_const, samples, cert_pts, x,
                                                        alpha, epsilon, output, 'rbf', sigma)))
    Y, X = [], []
    for r in res:
        res = r.get()
        if res is not None and res[0] is not None:
            Y.append(res[0])
            X.append(res[1])
    return np.vstack(X), np.vstack(Y)


class ConstraintDataset(Dataset):
    def __init__(self, decision_vars, output):
        self.x = torch.from_numpy(decision_vars).float()
        self.y = torch.from_numpy(output).float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx, :], self.y[idx, :]


def visualize_model(model, sol, y_sol, dim, X, Y, mode="GP", visualization='combo', **kwargs):
    print(sol, y_sol)
    low = sol - 2
    high = sol + 2
    steps = 0.1 * np.ones(dim)
    xx = create_decision_vars(low, high, steps)
    if mode == "GP":
        pred = model(torch.Tensor(xx)).mean.detach().numpy()
    elif mode == "NN":
        pred = model(torch.Tensor(xx)).detach().numpy()
    elif mode == "KRR":
        pred = model(xx)
    else:
        raise ValueError

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    if visualization == "combo":
        # Plot expectation
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_trisurf(xx[:, 0], xx[:, 1], pred[:, 0], alpha=0.5, cmap='viridis', edgecolor='none')
        ax.scatter(sol[0], sol[1], y_sol[0], marker='x', c='red')
        ax.scatter(X[:, 0], X[:, 1], Y[:, 0])

        # Plot norm
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_trisurf(xx[:, 0], xx[:, 1], pred[:, 1], alpha=0.5, cmap='viridis', edgecolor='none')
        ax.scatter(sol[0], sol[1], y_sol[1], marker='x', c='red')
        ax.scatter(X[:, 0], X[:, 1], Y[:, 1])
    elif visualization == "risk":
        eps = kwargs['epsilon']
        risk_pred = pred[:, 0] + eps * pred[:, 1]
        risk_sol = y_sol[0] + eps * y_sol[1]
        risk_train = Y[:, 0] + eps * Y[:, 1]
        ax = plt.axes(projection='3d')
        ax.plot_trisurf(xx[:, 0], xx[:, 1], risk_pred, alpha=0.5, cmap='viridis', edgecolor='none')
        ax.scatter(sol[0], sol[1], risk_sol, marker='x', c='red')
        ax.scatter(X[:, 0], X[:, 1], risk_train)
    plt.show()


def train_model(dataset, model_params, batch_size=50, epochs=50,
                lr=1e-3, model_type='NN', likelihood=None, **kwargs):
    """
    Train/fit ML model to the constraint dataset.

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
    model: object
        ML model
    batch_size: int
    epochs: int
    lr: float
    model_type: string
    likelihood: gpytorch.likelihoods
        Only used for the GP
    kwargs: dict
    """
    if model_type == "KRR":
        model = KRR(**model_params)
        model.fit(dataset.x.numpy(), dataset.y.numpy())
        return model
    dataloader = DataLoader(dataset, batch_size=batch_size)
    if model_type == "NN":
        model = ShallowNetwork(**model_params)
        model.train()
        loss_fn = nn.L1Loss()
    elif model_type == "GP":
        model = ExactGPModel(**model_params)
        loss_fn = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        likelihood.train()
    else:
        raise ValueError
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for epoch in range(epochs):
        if model_type == "NN":
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)

                optimizer.zero_grad()
                pred = model(X)
                # Compute prediction error
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()

            if epoch % 100 == 0:
                loss = loss.item()
                print(f"loss: {loss:>7f}  [{epoch:>5d}/{epochs:>5d}]")
        elif model_type == "GP":
            optimizer.zero_grad()
            pred = model(model.train_x)
            loss = -loss_fn(pred, model.train_y)
            loss.backward()
            optimizer.step()
            # l, _ = median_heuristic(model.train_x.numpy(), model.train_x.numpy())
            # for k in model.covar_module.sub_kernels():
            #     if k.has_lengthscale:
            #         k.initialize(lengthscale=l)
            # Debug information during training
            l = []
            for k in model.covar_module.sub_kernels():
                if k.has_lengthscale:
                    l.append(k.lengthscale.item())
            print('Iter %d/%d - Loss: %.3f   noise: %.3f   lengthscale: ' % (
                epoch + 1, epochs, loss.item(),
                model.likelihood.noise.item())
                  , l, "   mean: ", [mean.constant.item() for mean in model.mean_module.base_means])
        else:
            raise ValueError
    model.eval()
    return model


class KRR(object):
    """
    Kernel Ridge Regression model.
    """
    def __init__(self, kernel, sigma, in_dim, out_dim, alpha=1):
        """
        Create KRR object.

        Parameters
        ----------
        kernel: string
        sigma: float
        in_dim: int
        out_dim: int
        alpha: float
        """
        gamma = 1 / (2 * sigma**2)
        self.krr = KernelRidge(kernel=kernel,
                               gamma=gamma,
                               alpha=alpha)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fitted = False

    def fit(self, X, y):
        X_train, y_train = X, y
        if type(X) == torch.Tensor:
            X_train = X.numpy()
        if type(y) == torch.Tensor:
            y_train = y.numpy()
        self.krr.fit(X_train, y_train)
        # self.dual_coef = self.krr.dual_coef_
        self.train_x = self.krr.X_fit_
        self.fitted = True

    def __call__(self, X):
        if not self.fitted:
            raise AssertionError("Model not fitted")
        if type(X) == torch.Tensor:
            return self.krr.predict(X.numpy())
        elif type(X) == np.ndarray:
            return self.krr.predict(X)
        # elif type(X) == cas.MX:
        #     K = rbf_casadi(X, self.krr.X_fit_, gamma=self.krr.gamma)
        #     return cas.mtimes(K, self.dual_coef)
        else:
            raise ValueError


class ShallowNetwork(nn.Module):
    """
    Neural Network model for constraint learning.
    """
    def __init__(self, in_dim, out_dim, train_x, train_y):
        super(ShallowNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.train_x = train_x
        self.train_y = train_y
        self.elu_stack = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x):
        coef = self.elu_stack(x)
        return coef


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Gaussian Process model for constraint learning.
    """
    def __init__(self, train_x, train_y, in_dim, out_dim, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(prior=gpytorch.priors.NormalPrior(1, 0.01)),
            num_tasks=out_dim
        )
        for mean in self.mean_module.base_means:
            mean.initialize(constant=1)
            # mean.requires_grad = False
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=out_dim,
            rank=1
        )
        l, _ = median_heuristic(train_x.numpy(), train_x.numpy())
        for k in self.covar_module.sub_kernels():
            if k.has_lengthscale:
                k.initialize(lengthscale=l)
                # k.requires_grad = False
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def cvx_training_data(f_constraint, samples, cert_points, x, kernel,
                      kernel_param, alpha, epsilon, output="coef"):
    """
    Create data set for constraint learning.

    Given constraint f(x, xi) and certification points (xi, zeta),
    we create a dataset by computing for various decision variables x
    the corresponding majorizing function g(xi). Then the dataset looks like
    {(x_i, g(xi)_i\}.

    Parameters
    ----------
    f_constraint: callable f(x, samples (ndarray))
    samples: ndarray
        Samples.
    cert_points: ndarray
        Additional certification points used together with samples.
    alpha: float

    Returns
    -------
    dataset
    """
    n_samples, dim = samples.shape
    comb_points = np.vstack((cert_points, samples))
    n_cert = cert_points.shape[0]
    # Define variables
    coef = cp.Variable(n_samples + n_cert, name='coefficients')
    coef.value = np.zeros(n_samples + n_cert)

    # Define constraints
    f_const = np.around(f_constraint(x, comb_points), 6)  # This does not depend on decision variable
    ind_func = np.asarray(f_const > 0, dtype=float)
    g_rkhs = rkhs_func(coef, comb_points, comb_points, kernel=kernel, kernel_param=kernel_param)
    Eg_rkhs = rkhs_func_exp(coef, comb_points, samples, kernel=kernel, kernel_param=kernel_param)
    g_norm = rkhs_norm(coef, comb_points, kernel=kernel, kernel_param=kernel_param)
    constraints = [
        Eg_rkhs + epsilon * g_norm <= alpha,
        g_rkhs >= ind_func
    ]

    # Create objective
    # obj = cp.Minimize(cp.norm(g_rkhs - ind_func, 2))
    # obj = cp.Minimize(g_norm)
    obj = cp.Minimize(Eg_rkhs + epsilon * g_norm)

    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver="MOSEK")
    except:
        return None, x
    if prob.status == "infeasible":
        return None, x
    else:
        if output == "coef":
            return coef.value, x
        elif output == "f_val":
            return np.array([Eg_rkhs.value, g_norm.value]), x
        else:
            raise ValueError


def cas_const_learn(objective, model, samples, Xtrain, cert_points, kernel,
                    kernel_param, alpha, epsilon, constraint, output='coef', mode="NN"):
    """
    Solve approximate problem with learnt constraint using casadi.

    Parameters
    ----------
    Xtrain: ndarray
    cert_points: ndarray
    sigma: float
    alpha: float
    epsilon: float

    Returns
    -------
    x: ndarray
    """
    # Combine certification points and samples
    n_samples, dim = samples.shape
    comb_points = np.vstack((cert_points, samples))
    n_cert = cert_points.shape[0]

    # Compute Kernel Gram matrix
    K = rbf_kernel(comb_points, comb_points, kernel=kernel, kernel_param=kernel_param)
    K_sym = (K.T + K)/2 + np.eye(K.shape[0]) * 10e-5
    assert np.all(K_sym.T == K_sym), "Kernel matrix is not symmetric"
    L = np.linalg.cholesky(K_sym)
    if mode == "NN":
        # Create casadi NeuralNetwork
        cas_model = NNCasadi(name='constraint_ANN',
                             model=model,
                             opts={"enable_fd": True})
    elif mode == "GP":
        cas_model = GPCasadi(name='constraint_GP',
                             model=model,
                             likelihood=model.likelihood,
                             opts={"enable_fd": True})
    elif mode == "KRR":
        cas_model = KRRCasadi(name='constraint_KRR',
                              model=model,
                              opts={"enable_fd": True})

    # Setup Opti problem
    opt = cas.Opti()
    opt.solver('ipopt',
               {'print_time': False},
               {'print_level': 1, 'max_iter': 6000, "hessian_approximation": "limited-memory"},
               )
    x = opt.variable(model.in_dim, 1)
    opt.minimize(objective(x))
    if output == 'coef':
        pred = cas_model(x)
        emp_val = cas.sum1(K[:n_samples, :] @ pred.T) / (n_samples + n_cert)
        norm_reg = cas.norm_2(L @ pred.T)
        g_val = K @ pred.T
        opt.subject_to(g_val >= 0)
    elif output == 'f_val':
        pred = cas_model(x)
        emp_val, norm_reg = pred[0], pred[1]
        # norm_reg = cas_model(x)[1]
        opt.subject_to(emp_val >= 0)
        opt.subject_to(norm_reg >= 0)
    else:
        raise ValueError
    opt.subject_to(emp_val + epsilon * norm_reg <= alpha)
    opt.subject_to(x >= 0)

    # Add Trust-Region constraint
    q_hull = ConvexHull(Xtrain)
    A = q_hull.equations[:, :-1]
    b = q_hull.equations[:, -1]
    opt.subject_to(A @ x + b < 0)

    # Callback for debugging
    def cas_cb(iteration):
        db = opt.debug
        f_val = db.value(constraint(x, samples))
        n_satisfy = np.sum(f_val <= 0)
        print("Casadi internal iteration: {}".format(iteration))
        print("Decision variable: {}".format(db.value(x)))
        print("Expectation output: {}".format(db.value(emp_val)))
        print("RKHS norm: {}".format(db.value(norm_reg)))
        print("Constraint satifaction on samples: {}/{}".format(n_satisfy, samples.shape[0]))

    opt.callback(cas_cb)

    # Solve problem
    sol = opt.solve()
    return sol.value(x)

