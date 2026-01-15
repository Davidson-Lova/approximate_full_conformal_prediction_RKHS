"""
score function
"""

import numpy as np
import scipy.special as spsp


# From Stackoverflows: Avoiding overflow in log(cosh(x))
def logcosh(x):
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)


def sub(targets, predictions):
    return targets - predictions


def absolute_maker():
    def absolute_score(targets, predictions):
        resid = sub(targets, predictions)
        return np.abs(resid)

    score_lams = {"rho": 1, "reg": "C0"}

    return {
        "f": absolute_score,
        "lams": score_lams,
    }


def quadratic_maker(dmax=5):
    def quadratic_score(targets, predictions):
        resid = sub(targets, predictions)
        return resid**2

    def diff_quadratic_score(targets, predictions):
        resid = sub(targets, predictions)
        return -2 * resid

    def diff2_quadratic_score(targets, predictions):
        resid = sub(targets, predictions)
        return 2 * np.ones(resid.shape)

    score_lams = {
        "beta": 2,
        "xi": 0,
        "eta": 2,
        "rho": 2 * 2 * dmax,  # This is a problematic one
        "reg": "C2",
    }

    return {
        "f": quadratic_score,
        "df": diff_quadratic_score,
        "ddf": diff2_quadratic_score,
        "lams": score_lams,
    }


def huber_score(targets, predictions, alpha=1):
    resid = sub(targets, predictions)
    return spsp.huber(alpha, resid)


def pseudo_huber_maker(alpha=1.0, dmax=5):

    def pseudo_huber(x):
        return (alpha**2) * (np.sqrt(np.power(x / alpha, 2) + 1) - 1)

    def pseudo_huber_score(targets, predictions):
        resid = sub(targets, predictions)
        return pseudo_huber(resid)

    def diff_pseudo_huber(x):
        return x / np.sqrt(np.power(x / alpha, 2) + 1)

    def diff_pseudo_huber_score(targets, predictions):
        resid = sub(targets, predictions)
        return -diff_pseudo_huber(resid)

    def diff2_pseudo_huber(x):
        return 1 / np.power(np.power(x / alpha, 2) + 1, 3 / 2)

    def diff2_pseudo_huber_score(targets, predictions):
        resid = sub(targets, predictions)
        return diff2_pseudo_huber(resid)

    score_lams = {
        "beta": 1,
        "xi": (1.5 * ((4 / 5) ** 2.5)) * (alpha**-1),
        "eta": diff2_pseudo_huber(dmax),
        "rho": 1,
        "reg": "C2",
    }

    return {
        "f": pseudo_huber_score,
        "df": diff_pseudo_huber_score,
        "ddf": diff2_pseudo_huber_score,
        "lams": score_lams,
    }


def log_cosh_maker(alpha=1, dmax=5):
    def log_cosh_score(targets, predictions):
        resid = sub(targets, predictions)
        return alpha * logcosh(resid / alpha)

    def diff_log_cosh_score(targets, predictions):
        resid = sub(targets, predictions)
        return -np.tanh(resid / alpha)

    def diff2_log_cosh_score(targets, predictions):
        resid = sub(targets, predictions)
        res = np.exp(-2 * logcosh(resid / alpha)) / alpha
        return res

    def diff3_log_cosh(x):
        return -2 * np.tanh(x / alpha) * np.exp(-2 * logcosh(x / alpha))

    c = -np.arcsinh(np.sqrt(2) ** -1) * alpha

    score_lams = {
        "beta": 1 / alpha,
        "xi": diff3_log_cosh(c),
        "eta": diff2_log_cosh_score(dmax, 0),
        "rho": 1,
        "reg": "C2",
    }
    return {
        "f": log_cosh_score,
        "df": diff_log_cosh_score,
        "ddf": diff2_log_cosh_score,
        "lams": score_lams,
    }


def pinball_maker(tau=0.5):
    def pinball(x):
        return tau * (x >= 0) * x - (1 - tau) * (x < 0) * x

    def pinball_score(targets, predictions):
        return pinball(sub(targets, predictions))

    return pinball_score


def smoothed_pinball_maker(alpha=1.0, tau=0.5, dmax=5):
    def smoothed_pinball(x):
        return -alpha * spsp.log_expit(x / alpha) + tau * x

    def smoothed_pinball_score(targets, predictions):
        resid = sub(targets, predictions)
        return smoothed_pinball(resid)

    def diff_smoothed_pinball(x):
        return spsp.expit(-x / alpha) - tau

    def diff_smoothed_pinball_score(targets, predictions):
        resid = sub(targets, predictions)
        return diff_smoothed_pinball(resid)

    def diff2_smoothed_pinball(x):
        return (spsp.expit(x / alpha) * spsp.expit(-x / alpha)) / alpha

    def diff2_smoothed_pinball_score(targets, predictions):
        resid = sub(targets, predictions)
        return diff2_smoothed_pinball(resid)

    c = 3**0.5 + 2

    score_lams = {
        "beta": 0.25 * alpha**-1,
        "xi": (c * (c - 1)) / (((c + 1) ** 3) * (alpha**2)),
        "eta": diff2_smoothed_pinball(dmax),
        "rho": max(tau, 1 - tau),
        "reg": "C2",
    }

    return {
        "f": smoothed_pinball_score,
        "df": diff_smoothed_pinball_score,
        "ddf": diff2_smoothed_pinball_score,
        "lams": score_lams,
    }


def linex_maker(alpha=1):
    def linex(x):
        return np.exp(alpha * x) - alpha * x - 1

    def linex_score(targets, predictions):
        resid = sub(targets, predictions)
        return linex(resid)

    def diff_linex(x):
        return alpha * (np.exp(alpha * x) - 1)

    def diff_linex_score(targets, predictions):
        resid = sub(targets, predictions)
        return -diff_linex(alpha, resid)

    def diff2_linex(x):
        return (alpha**2) * np.exp(alpha * x)

    def diff2_linex_score(targets, predictions):
        resid = sub(targets, predictions)
        return -diff2_linex(resid)

    return {"f": linex_score, "df": diff_linex_score, "ddf": diff2_linex_score}


class non_conformity:
    def __init__(self):
        pass

    def maker(self, name):
        if name == "absolute":
            return absolute_maker
        elif name == "quadratic":
            return quadratic_maker
        elif name == "pseudo_huber":
            return pseudo_huber_maker
        elif name == "log_cosh":
            return log_cosh_maker
        elif name == "pinball":
            return pinball_maker
        elif name == "smoothed_pinball":
            return smoothed_pinball_maker
        elif name == "linex":
            return linex_maker
        else:
            print("Not available")
            return quadratic_maker
