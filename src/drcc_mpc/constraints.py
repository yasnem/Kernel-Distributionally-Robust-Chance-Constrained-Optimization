"""
Constraint implementation, linear, nonlinear and diff'able
"""
from casadi import MX, Function, exp, norm_2, vcat, hcat, sqrt, mtimes, power
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn import svm
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from matplotlib import colors
from drccp_utils.rkhs_utils import median_heuristic

class BaseConstraint(metaclass=ABCMeta):
    pass


class DifferentiableConstraint(object):
    def __init__(self, casadi_func):
        assert casadi_func.size_in(0) == casadi_func.size_in(1), "Z and E must have same dimensions!"
        assert casadi_func.n_out() == 1, "Should only have one output value."
        assert casadi_func.size_out(0) == (1, 1), "Constraint function should be mapped onto R"
        self.constraint_func = casadi_func
        self.jacobian = casadi_func.jacobian()
        self.dim = max(casadi_func.size_in(0))

    def eval(self, z, e):
        """
        Evaluate constraint function.

        Evaluate constraint function for a combination of z with different e,
        or just a tuple (z, e).

        Parameters
        ----------
        z: ndarray
            A (self.dim, 1) nominal state array.
        e: ndarray
            A (self.dim, 1) or (self.dim, n) error state array

        Returns
        -------
        c_func: ndarray
            A (n_out, n) array containing function evaluations.
        """
        c_func = []
        for i in range(e.shape[0]):
            c_func.append(self.constraint_func(z, e[i, :]))
        return vcat(c_func)

    def eval_gradient(self, z, e):
        """
        Evaluate the gradient wrt e of the constraint function.

        Parameters
        ----------
        z: ndarray
            A (self.dim, 1) array with the nominal state z.
        e: ndarray
            A (self.dim, n) arry with error states.

        Returns
        -------
        jac: ndarray
            A (self.dim, 1) or (self.dim, n) array containing the jacobian
            evaluated for z, e.
        """
        c_func = self.eval(z, e)
        jac = np.zeros((self.dim, e.shape[1]))
        for i in range(e.shape[1]):
            jac[:, i] = self.jacobian(z, e, c_func[:, i])[self.dim:]
        return jac


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
    return X, Y


def train_svm(c, gamma, X, Y):
    clf = svm.SVC(C=c, gamma=gamma)
    clf.fit(X, Y)
    return clf


def predict_svm(X, clf):
    dim = clf.support_vectors_.shape[1]
    print("# supp vectors: ", clf.support_vectors_.shape[0])
    ki = rbf_kernel(X.reshape((-1, dim)), clf.support_vectors_, gamma=clf.gamma)
    y_pred = clf.dual_coef_ @ ki.T + clf.intercept_
    return -y_pred


def plot_svm(clf, center, size, X_train, Y_train):
    """
    Plot trained SVM.

    Parameters
    ----------
    clf: sklearn.svm.SVC
    center: ndarray
    size: ndarray
    X_train: ndarray
    Y_train: ndarray
    """
    xx, yy = np.meshgrid(np.linspace(center[0] - size[0],
                                     center[0] + size[0], 500),
                         np.linspace(center[1] - size[1],
                                     center[1] + size[1], 500))

    Z = predict_svm(np.c_[xx.ravel(), yy.ravel()], clf)
    Z = Z.reshape(xx.shape)
    norm = colors.DivergingNorm(vmin=np.min(Z), vcenter=0, vmax=np.max(Z))

    im = plt.imshow(Z, interpolation='nearest',
                    extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
                    origin='lower', cmap=plt.cm.PuOr_r, norm=norm)

    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linestyles='dashed')
    plt.colorbar(im)
    plt.scatter(X_train[:, 0], X_train[:, 1], s=30, c=Y_train,
                cmap=plt.cm.Paired, edgecolors='k')
    plt.show()


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


def svm_constraint(center, size, n_points, plot=False, gamma=None):
    X, Y = make_constraint_data(center, size, n_points)
    if gamma is None:
        _, gamma = median_heuristic(X, X)
    clf = train_svm(10, gamma, X, Y)
    if plot:
        plot_svm(clf, center, size, X, Y)
    return casadi_svm(clf), clf, X, Y


def svm_upperbound(clf):
    ki = rbf_kernel(clf.support_vectors_, clf.support_vectors_, gamma=clf.gamma)
    upperbound = np.sqrt(clf.dual_coef_ @ ki @ clf.dual_coef_.T)
    return upperbound[0, 0]


def compute_f(a, f0, K):
    f_e = []
    for j in range(a.shape[0]):
        tmp = []
        for i in range(K.shape[0]):
            tmp.append(mtimes(a[j, :], K[i, :]) + f0[j])
        f_e.append(hcat(tmp))
    f_e = vcat(f_e)
    cas_rkhsf = Function('rkhs_fun', [a, f0], [f_e])
    return cas_rkhsf(a, f0)


def compute_rkhsnorm_squared(a, K):
    rkhs_norm_sq_vec = []
    for i in range(a.shape[0]):
        rkhs_norm_sq_vec.append(mtimes(mtimes(a[i, :], K), a[i, :].T))
    rkhs_norm_sq = vcat(rkhs_norm_sq_vec)
    cas_rkhsnorm = Function('rkhs_norm_squared', [a], [rkhs_norm_sq])
    return cas_rkhsnorm(a)


def compute_rkhsnorm(a, K):
    cas_rkhsnorm = Function('rkhs_norm', [a], [sqrt(mtimes(mtimes(a, K), a.T))])
    return cas_rkhsnorm(a)


def casadi_rkhsfun(a, f0, gamma, sup_vec_shape):
    # TODO(yassine): Avoid this for now. Need to switch back to a kernel function argument and fix the casadi method.
    # given a casadi kernel
    # given sup vectors of shape (n_sv, dim)
    # a is an optimization variable (MX vector) of the form (n_out, n_sv)
    dim = sup_vec_shape[1]
    e = MX.sym('e', dim)
    sv = MX.sym('sv', sup_vec_shape[0], dim)
    ki = []
    for i in range(sup_vec_shape[0]):
        ki.append(power(exp(-gamma * norm_2(e - sv[i, :].T)), 2))
    # f_e = mtimes(a, vcat(ki)) + f0
    K = vcat(ki)
    K = np.ones(sup_vec_shape[0])/sup_vec_shape[0]
    f_e = mtimes(a, K) + f0
    # f_e has shape (n_out, 1)
    cas_rkhsf = Function('rkhs_fun', [e, sv, f0, a], [f_e])
    return cas_rkhsf


def compute_rkhs_norm(a, gamma, sup_vec):
    # TODO(yassine): Avoid this for now. Need to switch back to a kernel function argument and fix the casadi method.
    """
    Compute the rkhs norm of a rkhs function.

    F(x) = a.T * K(sv, x)

    Parameters
    ----------
    a: ndarray
        A (n_sv, 1) array with coefficients
    casadi_kernel: casadi.Function or callable
    sup_vec: ndarray
        A (n_sv, xdim) array with the support vectors.

    Returns
    -------
    rkhs_norm: float
    """
    K = []
    for i in range(sup_vec.shape[0]):
        K.append([])
        for j in range(0, sup_vec.shape[0]):
            K[-1].append(power(exp(-gamma * norm_2(sup_vec[i, :] - sup_vec[j, :])), 2))
        K[-1] = hcat(K[-1])
    K = vcat(K)
    norm = []
    for i in range(a.shape[0]):
        norm.append(mtimes(mtimes(a[i, :], K), a[i, :].T))
    return sqrt(vcat(norm))


def ellipsoid_constraint(center, axis):
    """
    Create ellipsoid constraint.

    Parameters
    ----------
    center: ndarray
    axis: ndarray

    Returns
    -------
    ellipsoid_c: casadi.Function
    """
    dim_x = len(center)
    z = MX.sym('z', dim_x)
    e = MX.sym('e', dim_x)
    tmp_lst = []
    for i in range(dim_x):
        tmp_lst.append(((z[i] + e[i] - center[i])/axis[i])**2)
    ellipsoid_c = Function('e_c', [z, e], [sum(tmp_lst) - 1])
    return ellipsoid_c


def exponential_constraint(x_lvl, growth_rate):
    z = MX.sym('z', 2)
    e = MX.sym('e', 2)
    exp_c = Function('exp_c', [z, e],
                     [-1 + x_lvl + exp(growth_rate* (z[0] + e[0])) - z[1] - e[1]])
    return exp_c
