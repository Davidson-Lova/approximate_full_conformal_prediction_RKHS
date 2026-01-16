"""
Usefull function
"""

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
