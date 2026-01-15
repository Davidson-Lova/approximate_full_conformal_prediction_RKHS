"""
Loss functions
"""

import numpy as np
import scipy.special as spsp


def sub(targets, predictions):
    return targets - predictions


def huber_loss(targets, predictions, alpha=1):
    resid = sub(targets, predictions)
    return spsp.huber(alpha, resid)


def absolute_maker():
    def absolute_loss(targets, predictions):
        resid = sub(targets, predictions)
        return np.abs(resid)

    loss_lams = {"rho": 1}

    return {
        "f": absolute_loss,
        "lams": loss_lams,
    }


def quadratic_maker(dmax=5.0):
    def quadratic_loss(targets, predictions):
        resid = sub(targets, predictions)
        return resid**2

    def diff_quadratic_loss(targets, predictions):
        resid = sub(targets, predictions)
        return -2 * resid

    def diff2_quadratic_loss(targets, predictions):
        resid = sub(targets, predictions)
        return 2 * np.ones(resid.shape)

    loss_lams = {
        "beta": 2,
        "xi": 0,
        "eta": 2,
        "rho": 2 * 2 * dmax,
    }

    return {
        "f": quadratic_loss,
        "df": diff_quadratic_loss,
        "ddf": diff2_quadratic_loss,
        "lams": loss_lams,
    }


def pseudo_huber_maker(alpha=1.0):

    def pseudo_huber(x):
        return (alpha**2) * (np.sqrt(np.power(x / alpha, 2) + 1) - 1)

    def pseudo_huber_loss(targets, predictions):
        resid = sub(targets, predictions)
        return pseudo_huber(resid)

    def diff_pseudo_huber(x):
        return x / np.sqrt(np.power(x / alpha, 2) + 1)

    def diff_pseudo_huber_loss(targets, predictions):
        resid = sub(targets, predictions)
        return -diff_pseudo_huber(resid)

    def diff2_pseudo_huber(x):
        return 1 / np.power(np.power(x / alpha, 2) + 1, 3 / 2)

    def diff2_pseudo_huber_loss(targets, predictions):
        resid = sub(targets, predictions)
        return diff2_pseudo_huber(resid)

    loss_lams = {
        "beta": 1,
        "xi": (1.5 * ((4 / 5) ** 2.5)) * (alpha**-1),
        "eta": 0,
        "rho": alpha,
    }

    return {
        "f": pseudo_huber_loss,
        "df": diff_pseudo_huber_loss,
        "ddf": diff2_pseudo_huber_loss,
        "lams": loss_lams,
    }


# From Stackoverflows: Avoiding overflow in log(cosh(x))
def logcosh(x):
    s = np.sign(x) * x
    p = np.exp(-2 * s)
    return s + np.log1p(p) - np.log(2)


def sech2(x):
    return np.exp(-2 * logcosh(x))


def log_cosh_maker(alpha=1):
    def log_cosh_loss(targets, predictions):
        resid = sub(targets, predictions)
        return alpha * logcosh(resid / alpha)

    def diff_log_cosh_loss(targets, predictions):
        resid = sub(targets, predictions)
        return -np.tanh(resid / alpha)

    def diff2_log_cosh_loss(targets, predictions):
        resid = sub(targets, predictions)
        res = np.exp(-2 * logcosh(resid / alpha)) / alpha
        return res

    c = -np.arcsinh(np.sqrt(2) ** -1) * alpha

    loss_lams = {
        "beta": 1 / alpha,
        "xi": 2
        * np.tanh(np.arcsinh(np.sqrt(2) ** -1))
        * sech2(np.arcsinh(np.sqrt(2) ** -1))
        / alpha**2,
        "eta": 0,
        "rho": 1,
    }

    return {
        "f": log_cosh_loss,
        "df": diff_log_cosh_loss,
        "ddf": diff2_log_cosh_loss,
        "lams": loss_lams,
    }


def pinball_maker(tau=0.5):
    def pinball(x):
        return tau * (x >= 0) * x - (1 - tau) * (x < 0) * x

    def pinball_loss(targets, predictions):
        return pinball(sub(targets, predictions))

    return pinball_loss


def smoothed_pinball_maker(alpha=1.0, tau=0.5):
    def smoothed_pinball(x):
        return -alpha * spsp.log_expit(x / alpha) + tau * x

    def smoothed_pinball_loss(targets, predictions):
        resid = sub(targets, predictions)
        return smoothed_pinball(resid)

    def diff_smoothed_pinball(x):
        return spsp.expit(-x / alpha) - tau

    def diff_smoothed_pinball_loss(targets, predictions):
        resid = sub(targets, predictions)
        return diff_smoothed_pinball(resid)

    def diff2_smoothed_pinball(x):
        return (spsp.expit(x / alpha) * spsp.expit(-x / alpha)) / alpha

    def diff2_smoothed_pinball_loss(targets, predictions):
        resid = sub(targets, predictions)
        return diff2_smoothed_pinball(resid)

    c = 3**0.5 + 2

    loss_lams = {
        "beta": 0.25 * alpha**-1,
        "xi": (c * (c - 1)) / (((c + 1) ** 3) * (alpha**2)),
        "eta": 0,
        "rho": max(tau, 1 - tau),
    }

    return {
        "f": smoothed_pinball_loss,
        "df": diff_smoothed_pinball_loss,
        "ddf": diff2_smoothed_pinball_loss,
        "lams": loss_lams,
    }


def linex_maker(alpha=1):
    def linex(x):
        return np.exp(alpha * x) - alpha * x - 1

    def linex_loss(targets, predictions):
        resid = sub(targets, predictions)
        return linex(resid)

    def diff_linex(x):
        return alpha * (np.exp(alpha * x) - 1)

    def diff_linex_loss(targets, predictions):
        resid = sub(targets, predictions)
        return -diff_linex(alpha, resid)

    def diff2_linex(x):
        return (alpha**2) * np.exp(alpha * x)

    def diff2_linex_loss(targets, predictions):
        resid = sub(targets, predictions)
        return -diff2_linex(resid)

    return {"f": linex_loss, "df": diff_linex_loss, "ddf": diff2_linex_loss}


def maker(name):
    if name == "absolute":
        return absolute_maker
    if name == "quadratic":
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
        raise ValueError("Function not found")
