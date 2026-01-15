"""
Usefull function
"""

import numpy as np
import portion as P
from scipy.optimize import root_scalar


def inter_finder(objective, y_min, y_max, y_hat):
    """
    Computes the interval contains within (y_min, y_max)
    where the objective function is non-negative

    Args:
        objective (callable): objective function
        y_min (float): output sample minimum
        y_max (float): output sample maximum
        y_hat (float): prediction

    Returns:
        (P.open): interval where the objective function is non-negative

    """
    lb = y_min
    ub = y_max

    fy_min = objective(y_min)
    fy_max = objective(y_max)
    fy_hat = objective(y_hat)
    fy_0 = objective(0)

    if fy_min * fy_max < 0:
        mid_finder = root_scalar(objective, bracket=[y_min, y_max])
        if fy_min < 0:
            lb = mid_finder.root
        else:
            ub = mid_finder.root
        return P.closed(lb, ub)
    else:
        if fy_min >= 0:
            if fy_min * fy_0 < 0:
                mid_finder_left = root_scalar(objective, bracket=[y_min, 0])
                mid_finder_right = root_scalar(objective, bracket=[0, y_max])
                return P.closed(lb, mid_finder_left.root) | P.closed(
                    mid_finder_right.root, ub
                )
            else:
                return P.closed(lb, ub)
        else:
            if fy_min * fy_hat < 0:
                lb_finder = root_scalar(objective, bracket=[y_min, y_hat])
                ub_finder = root_scalar(objective, bracket=[y_hat, y_max])
                return P.closed(lb_finder.root, ub_finder.root)
            else:
                print("Prediction not in the predictive region")
                return P.open(0, 0)


# predictor
def solveh_im(M, b, rtol=None):
    """
    Solves a linear equation in the image

    Args:
        M (np.array): matrix
        b (np.array): vector
        rtol (np.float): tolerance level

    Returns:
        (np.array): M*b
    """
    U, S, Vh = np.linalg.svd(M)
    if rtol is None:
        rtol = S.max(-1) * np.max(M.shape[-2:]).astype(S.dtype) * np.finfo(S.dtype).eps
    S = S[S > rtol]

    # Compute rank
    trace = np.sum(np.diagonal(M))
    r = np.sum(np.cumsum(S) <= trace)

    x_im_A = U[:, :r] @ ((Vh[:r, :] @ b) / S[:r].reshape(-1, 1))

    return x_im_A


# smallest eigen value
def smallest_non_zero_eig(M, rtol=None):
    """
    Computes the smallest non-zero eigen-value

    Args:
        M (np.array): matrix

    Returns:
        (np.float): smallest non-zero eigen-value
    """
    _, S, _ = np.linalg.svd(M)
    if rtol is None:
        rtol = S.max(-1) * np.max(M.shape[-2:]).astype(S.dtype) * np.finfo(S.dtype).eps
    S = S[S > rtol]

    # Compute rank
    trace = np.sum(np.diagonal(M))
    up_to_trace = np.cumsum(S)
    r = np.sum(up_to_trace <= trace)

    #
    nonzero_sing_values = S[:r]
    if len(nonzero_sing_values) == 0:
        raise ValueError("All sing_values are zero or close to zero")

    return np.min(nonzero_sing_values)


def min_loc(f1, f2):
    if callable(f1) and callable(f2):
        return lambda a: np.minimum(f1(a), f2(a))
    elif callable(f1) and not callable(f2):
        return lambda a: np.minimum(f1(a), f2)
    elif not callable(f1) and callable(f2):
        return lambda a: np.minimum(f1, f2(a))
    else:
        return lambda a: np.minimum(f1, f2)


def add(f1, f2):
    if callable(f1) and callable(f2):
        return lambda a: f1(a) + f2(a)
    elif callable(f1) and not callable(f2):
        return lambda a: f1(a) + f2
    elif not callable(f1) and callable(f2):
        return lambda a: f1 + f2(a)
    else:
        return lambda a: f1 + f2


def sub(f1, f2):
    if callable(f1) and callable(f2):
        return lambda a: f1(a) - f2(a)
    elif callable(f1) and not callable(f2):
        return lambda a: f1(a) - f2
    elif not callable(f1) and callable(f2):
        return lambda a: f1 - f2(a)
    else:
        return lambda a: f1 - f2


def mul(f1, f2):
    if callable(f1) and callable(f2):
        return lambda a: f1(a) * f2(a)
    elif callable(f1) and not callable(f2):
        return lambda a: f1(a) * f2
    elif not callable(f1) and callable(f2):
        return lambda a: f1 * f2(a)
    else:
        return lambda a: f1 * f2


def abs_callable(f):
    if callable(f):
        return lambda a: abs(f(a))
    else:
        return lambda a: abs(f)


def square_callable(f):
    if callable(f):
        return lambda a: f(a) ** 2
    else:
        return lambda a: f**2


def square_root_callable(f):
    if callable(f):
        return lambda a: f(a) ** (1 / 2)
    else:
        return lambda a: f ** (1 / 2)


def neg_callable(f):
    if callable(f):
        return lambda a: -f(a)
    else:
        return lambda a: -f


def partial_2(f, x):
    def res(y):
        return f(y, x)

    return res


def comp(f1, f2, x):
    def res(y):
        return f1(x, f2(y))

    return res


def comp_2(f1, f2):
    def res(y):
        return f1(y, f2(y))

    return res


def flat_l(l):
    return [xx for x in l for xx in x]


def interval_length(interval):
    """
    Compute the length of an interval

    Args:
        interval (P.closed): interval

    Returns:
        (float): interval length
    """
    if interval.empty:
        return 0
    length = 0
    for subinterval in interval:
        length += subinterval.upper - subinterval.lower
    return length


def p_value_maker(l_ncs):
    """
    Formulates a conformal p-value function
    from a list of non-conformity score functions

    Args:
        l_ncs (list[callable]): list of non-conformity score

    Returns:
        (callable): conformal p-value function
    """
    d_ncs = [[sub(fi, fs[-1]) for fi in fs[:-1]] for fs in l_ncs]

    def maker(lf):
        N_train = len(lf)

        def res(y):
            return (1 + sum([f(y) >= 0 for f in lf])) / (N_train + 1)

        return res

    return [maker(funcs) for funcs in d_ncs]


def region_maker(p_value_function, y_min, y_max, Y_hat_j):
    """
    Formulates conformal region

    Args:
        p_value_function (callable): conformal p-value function
        y_min (float): output sample minimum
        y_max (float): output sample maximum
        Y_hat_j (float): prediction

    Returns:
        (callable): region predictor as a function of the confidence control level
    """

    def res(epsilon):
        objective_function = [sub(f, epsilon) for f in p_value_function]
        return [
            inter_finder(objective, y_min, y_max, yhat)
            for objective, yhat in zip(objective_function, Y_hat_j)
        ]

    return res
