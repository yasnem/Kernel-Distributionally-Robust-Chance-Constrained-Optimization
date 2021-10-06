import sympy as sp
import cvxpy as cp
import numpy as np


sp_to_cp = {
    sp.exp: cp.exp,
    sp.log: cp.log
}


def get_monoms_list(p):
    """
    Given a sympy Polynomial expression p, compute a list with all monomials contained.

    Parameters
    ----------
    p: sympy.Poly
        A sympy Polynomial

    Returns
    -------
    monoms: list
        List with sympy monomials.
    """
    return [sp.prod(x**k for x, k in zip(p.gens, mon)) for mon in p.monoms()]


def parse(expr, var_dict):
    """
    Parse a polynomial expression in its appropriate base.

    Parameters
    ----------
    expr: Sympy experession
    var_dict: dict
        Dictionnary mapping sympy Variables to cvxpy Variables
    """
    # We reached a leave in the expression Tree
    # expr.is_symbol is necessary for MatrixSymbol Elements
    if expr.is_Atom or expr.is_symbol:
        if expr.func.is_Symbol:  # This is a scalar symbol
            return var_dict[expr]
        elif expr.func.is_symbol:  # This is a Matrix symbol
            return var_dict[expr.symbol][expr.indices]
        elif expr.func.is_number:
            return float(expr)
        else:
            raise ValueError
    else:
        expressions = []
        for arg in expr.args:
            expressions.append(parse(arg, var_dict))
        if expr.func.is_Add:
            cvx_expr = sum(expressions)
        elif expr.func.is_Mul:
            cvx_expr = np.prod(expressions)
        elif expr.func.is_Pow:
            assert len(expressions) == 2, 'There should only be 2 elements!'
            cvx_expr = expressions[0] ** expressions[1]
        elif expr.func in sp_to_cp:
            assert len(expressions) == 1, "There should only be one expression by now!"
            cvx_expr = sp_to_cp[expr.func](expressions[0])
        else:
            raise ValueError('The following expression {0} of type {1} is not supported.'.format(expr, expr.func))
        return cvx_expr


def parse_sym_to_cvx(polynomial, var_dict, basis):
    """
    Parse a sympy polynomial to a cvx expresion.

    Parameters
    ----------
    polynomial: sympy.Poly
    var_dict: dict
        Dictionary that contains mapping from cvxpy.Variable to sympy.Symbol
    basis: iteratable(sp.Symbol, ...)

    Returns
    -------
    poly_dict: dict
        keys -- monomials in sympy form
        values -- coefficients in cvxpy form
    m: list
        List of monomials in polynom
    """
    poly = polynomial.as_poly(basis)
    monoms = get_monoms_list(poly)
    poly_dict = {}
    for m in monoms:
        c = poly.coeff_monomial(m)
        poly_dict[m] = parse(expr=c, var_dict=var_dict)
    return poly_dict, set(monoms)


def build_sos_basis(target_monoms, x_arr, var_dict):
    """
    Build appropriate SOS basis for target polynomial.

    Parameters
    ----------
    target_monoms: list
        List with the target monomials (obtained from get_monoms_list)
    x_arr: np.ndarray
        Array with sympy Symbols of the variables.
    var_dict: dict
        Dictionary that contains mapping from cvxpy.Variable to sympy.Symbol

    Returns
    -------
    bsos: sympy.Matrix
        A sympy Vector containing all the necessary monomials
    Q_sym: sympy.MatrixSymbol
        Matrix symbol for the PSD matrix in the SOS decomposition
    var_dict: dict
        Added psd matrix mapping.
    """
    # Create target polynomial
    p = 0
    for ele in target_monoms:
        p += ele
    p = p.as_poly()

    # build target basis
    deg = int(np.ceil(p.degree() / 2))
    basis_poly = 0
    for dd in range(deg):
        basis_poly += np.sum(x_arr) ** (dd + 1)
    basis_poly = basis_poly.as_poly() + 1
    bsos_monoms = get_monoms_list(basis_poly)
    n_basis_sos = len(bsos_monoms)
    bsos = sp.Matrix(bsos_monoms)

    # Create PSD Matrxi according to found basis
    Q_sym = sp.MatrixSymbol("Q", n_basis_sos, n_basis_sos)
    Q_cvx = cp.Variable((n_basis_sos, n_basis_sos), symmetric=True)
    var_dict[Q_sym] = Q_cvx
    return bsos, Q_sym, var_dict


def create_sos_constraints(lhs_constraint, var_sym, var_arr, var_dict, eps=1e-8):
    """
    Create a list with cvx constraints.

    Parameters
    ----------
    lhs_constraint: sympy.Poly
        The lefthandside constraint we want to get a SOS decomposition of.
    var_sym: iteratable(sympy.Symbol, ...)
        Iteratable containing all the sympy Variables.
    var_arr: np.ndarray
        Numpy array with all the variable for matrix mulitplication.
    var_dict: dict
        Dictionary mapping sympy Variables to cvxpy Variables

    Returns
    -------
    constraints: list
    """
    lhs_dict, lhs_monoms = parse_sym_to_cvx(lhs_constraint, var_dict, var_sym)

    # construct an appropriate SOS basis
    bsos, Q_sym, var_dict = build_sos_basis(lhs_monoms, var_arr, var_dict)
    Q = sp.Matrix(Q_sym)
    n_sos = Q.shape[0]

    # SOS decomposition as the RHS of the constraint
    rhs = (bsos.transpose() @ Q @ bsos)[0, 0].as_poly(var_sym)
    rhs_dict, rhs_monoms = parse_sym_to_cvx(rhs, var_dict, var_sym)

    # Create cvxpy constraints
    constraints = [var_dict[Q_sym] >> eps * np.eye(n_sos)]
    assert len(lhs_dict.keys() - rhs_dict.keys()) == 0, "SOS decomposition is not powerful enough!"
    for m in rhs_monoms:
        if m not in lhs_dict:
            constraints.append(rhs_dict[m] == 0)
        else:
            constraints.append(lhs_dict[m] == rhs_dict[m])
    return constraints, Q_sym