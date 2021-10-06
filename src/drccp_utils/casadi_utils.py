import casadi as cas
import torch
import numpy as np


def cas_rkhs_func(f0, w, K):
    """
    Evaluate rkhs function of form f(x) = alpha.T * K(X, x).

    f0: integer-like -- offset of rkhs function
    w: array-like -- weights of rkhs function (n_samples, 1)
    K: ndarray -- Gram matrix
    """
    f = cas.mtimes(K, w) + f0
    return f


def cas_rkhs_func_exp(f0, w, K):
    """
    Evaluate rkhs function of form f(x) = alpha.T * K(X, x).

    f0: integer-like -- offset of rkhs function
    w: array-like -- weights of rkhs function (n_samples, 1)
    K: ndarray -- Gram matrix
    """
    K = K @ np.ones(K.shape[0]) / K.shape[0]
    f = cas.dot(w, K) + f0
    return f


def cas_rkhs_norm_squared(w, K):
    """
    alpha: ndarray -- weights of rkhs function
    K: ndarray -- Gram matrix
    param: dict
        sigma: float -- bandwidth of kernel
    """
    # L = cholesky_decomposition(K)
    # norm = cas.norm_2(cas.mtimes(L, w))
    quad_norm = cas.dot(w, cas.mtimes(K, w))
    return quad_norm


def rbf_casadi(X, Y=None, kernel='rbf', gamma=None):
    '''kernel comput. using casadi'''
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <rbf_kernel>`.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float, default None
        If None, defaults to 1.0 / n_features
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    nsx,nfx = X.shape
    nsy,nfy = Y.shape

    K = cas.MX(nsx, nsy)

    for i in range(nsx):
        for j in range(nsy):
            if kernel=='rbf':
                K[i,j] = cas.exp(-gamma * cas.sumsqr(X[i] - Y[j]))
            elif kernel=='sum':
                # use sum of kernels, still valid
                K[i, j] = cas.sum([cas.exp(-s*gamma * cas.sumsqr(X[i] - Y[j]))
                                for s in [0.001, 0.01, 0.1, 1.0, 10]])
    # K = euclidean_distances(X, Y, squared=True)
    # K *= -gamma
    # np.exp(K, K)  # exponentiate K in-place
    return K


class NNCasadi(cas.Callback):
    def __init__(self, name, model, opts={}):
        super(NNCasadi, self).__init__()
        self.model = model
        self.construct(name, opts)

    def get_n_in(self): return 2

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cas.Sparsity.dense(1, self.model.in_dim)

    def get_sparsity_out(self, i):
        return cas.Sparsity.dense(1, self.model.out_dim)

    def init(self):
        self.model.eval()

    @torch.no_grad()
    def eval(self, arg):
        pred = self.model(torch.from_numpy(np.array(arg[0] + arg[1])).float()).numpy()
        return [pred]


class KRRCasadi(cas.Callback):
    def __init__(self, name, model, opts={}):
        super(KRRCasadi, self).__init__()
        self.model = model
        self.construct(name, opts)

    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cas.Sparsity.dense(1, self.model.in_dim)

    def get_sparsity_out(self, i):
        return cas.Sparsity.dense(1, self.model.out_dim)

    @torch.no_grad()
    def eval(self, arg):
        pred = self.model(np.array(arg[0]))
        # pred = pred * self.range + self.shift
        return [pred]


class GPCasadi(cas.Callback):
    def __init__(self, name, model, likelihood, opts={}):
        super(GPCasadi, self).__init__()
        self.model = model
        self.likelihood = likelihood
        self.construct(name, opts)

    def get_n_in(self): return 1

    def get_n_out(self): return 1

    def get_sparsity_in(self, i):
        return cas.Sparsity.dense(1, self.model.in_dim)

    def get_sparsity_out(self, i):
        return cas.Sparsity.dense(1, self.model.out_dim)

    def init(self):
        self.model.eval()

    @torch.no_grad()
    def eval(self, arg):
        f_pred = self.model(torch.from_numpy(np.array(arg[0])).float())
        pred = self.likelihood(f_pred).mean.numpy()
        # pred = pred * self.range + self.shift
        return [pred]


class PytorchEvaluator(cas.Callback):
    def __init__(self, t_in, t_out, opts={"enable_fd": True}):
        """
          t_in: list of inputs (pytorch tensors)
          t_out: list of outputs (pytorch tensors)
        """
        cas.casadi.Callback.__init__(self)
        assert isinstance(t_in, list)
        self.t_in = t_in
        assert isinstance(t_out, list)
        self.t_out = t_out
        self.construct("PytorchEvaluator", opts)
        self.refs = []

    def get_n_in(self): return len(self.t_in)

    def get_n_out(self): return len(self.t_out)

    def get_sparsity_in(self, i):
        return cas.Sparsity.dense(*list(self.t_in[i].size()))

    def get_sparsity_out(self, i):
        return cas.Sparsity.dense(*list(self.t_out[i].size()))

    def eval(self, arg):
        # arg0 = torch.tensor(np.array(arg[0]).T, device=torch.device('cpu'), dtype=torch.float)
        return [cas.DM(arg0.detach().numpy()) for arg0 in self.t_out]

    # Vanilla tensorflow offers just the reverse mode AD
    def has_reverse(self, nadj): return nadj == 1

    def get_reverse(self, nadj, name, inames, onames, opts):
        # Construct tensorflow placeholders for the reverse seeds
        adj_seed = [torch.ones(self.sparsity_out(i).shape[0],
                               dtype=torch.float,
                               device=torch.device("cpu"),
                               requires_grad=True) for i in range(self.n_out())]

        # Create another TensorFlowEvaluator object
        for i, t_out in enumerate(self.t_out):
            t_out.backward(torch.ones(self.t_in[i].size()))

        out = [t_in.grad for t_in in self.t_in]

        # callback = PytorchEvaluator(self.t_in + adj_seed, self.t_out.backward())
        callback = PytorchEvaluator(self.t_in + adj_seed, out)
        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out+adj_seed form that CasADi expects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        adj_seed = self.mx_out()
        return cas.Function(name,
                            nominal_in + nominal_out + adj_seed,
                            callback.call(nominal_in + adj_seed),
                            inames, onames)
