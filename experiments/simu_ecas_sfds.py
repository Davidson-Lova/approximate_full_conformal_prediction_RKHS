"""
The present aim is to compare the rate of evolution of
the implicit upper bound on the thickness of the approximation regions
utilizing:
    + uniform stability bounds (approx_fcp_0),
    + local stability bounds (approx_fcp_1)
"""

# %%
# Useful packages
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("..")
import sklearn.datasets as datasets
from sklearn.preprocessing import StandardScaler

from predictive_region.approx_fcp_0 import approx_fcp_0
from predictive_region.approx_fcp_1 import approx_fcp_1
from predictive_region.approx_fcp_2 import approx_fcp_2
from predictive_region.oracle_cp import oracle_cp
from predictive_region.pred_models import kernel_regression
from predictive_region.scp import scp
from predictive_region.utils.kernels import kernels
from predictive_region.utils.losses import losses
from predictive_region.utils.non_conformity import non_conformity
from predictive_region.utils.utils import interval_length, p_value_maker, region_maker

# %%
methods = {
    "oracle_cp": oracle_cp,
    "scp": scp,
    "approx_fcp_0": approx_fcp_0,
    "approx_fcp_1": approx_fcp_1,
    "approx_fcp_2": approx_fcp_2,
}


def set_style():
    plt.rcParams["text.usetex"] = True
    font = {"family": "normal", "weight": "bold", "size": 22}
    plt.rc("font", **font)


set_style()


def param_reader(param_path):
    with open(param_path, "r") as file:
        params = json.load(file)
    return params


def load_data(dataset, n_samples, n_features):
    if dataset == "friedman1":
        X_, Y_ = datasets.make_friedman1(
            n_samples=n_samples, n_features=n_features, noise=1
        )
        Y_ = Y_.reshape(-1, 1)

    if dataset == "synthetic":
        dense = 0.7
        X_, Y_ = datasets.make_regression(
            n_samples=n_samples,
            n_features=n_features,
            # random_state=random_state,
            n_informative=int(n_features * dense),
            noise=1,
        )
        Y_ = Y_.reshape(-1, 1)

    return X_, Y_


def save_result(result, path):
    np.save(path, result)


def result_reader(path):
    return np.load(path, allow_pickle=True).tolist()


def evaluate(method, X_train, y_seen, X_test, z_test, param):
    if method.name == "approx_fcp_0":
        ncs, K_diag, hat_Y_Np1 = method._ncs_(
            X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
        )
        upper_bounds = method.thickness_bound_explicit(K_diag)
    elif method.name == "approx_fcp_1":
        (
            ncs,
            K_diag,
            hat_Y_Np1,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
        ) = method._ncs_(
            X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
        )
        upper_bounds = method.thickness_bound_explicit(K_diag)
    elif method.name == "approx_fcp_2":
        (
            ncs,
            K_diag,
            hat_Y_Np1,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
            K_diag_3half_mean,
        ) = method._ncs_(
            X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
        )
        upper_bounds = method.thickness_bound_explicit(K_diag)

    region = method.region(
        X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
    )
    region_up = region["up"]["region"](param["region"]["control_level"])
    region_lo = region["low"]["region"](param["region"]["control_level"])

    confidence_gaps = [
        interval_length(r_up) - interval_length(r_lo)
        for r_up, r_lo in zip(region_up, region_lo)
    ]

    return {"upper_bound": upper_bounds[0], "confidence_gap": confidence_gaps[0]}


def one_rep(i_rep, grid_n_samples, param, cp_methods):
    gain = {name: {col: [] for col in param["columns"]} for name in param["methods"]}

    for n_samples in grid_n_samples:
        print("rep N°{} - n_samples: {}".format(i_rep, n_samples))

        # sample data
        X_, y_ = load_data(param["dataset"], n_samples, param["n_features"])

        # format data
        X_train, X_test = X_[:-1, :], X_[-1, :].reshape(1, -1)
        y_seen, y_not_seen = y_[:-1, :], y_[-1, :].reshape(1, -1)
        z_test = np.zeros(y_not_seen.shape)

        # normalization
        X_scalar = StandardScaler()
        X_scalar.fit(X_train)
        X_train = X_scalar.transform(X_train)
        X_test = X_scalar.transform(X_test)

        y_scalar = StandardScaler()
        y_scalar.fit(y_seen)
        y_seen = y_scalar.transform(y_seen)
        y_not_seen = y_scalar.transform(y_not_seen)

        eval_data = {
            name: evaluate(cp_methods[name], X_train, y_seen, X_test, z_test, param)
            for name in param["methods"]
        }

        for name in param["methods"]:
            for col in param["columns"]:
                gain[name][col] += [eval_data[name][col]]

    return gain


# %%
def run(path_param, path_output):

    param = param_reader(path_param)
    grid_n_samples = np.unique(
        np.int64(
            np.exp2(
                np.linspace(
                    np.log2(param["n_min"]), np.log2(param["n_max"]), param["nb_steps"]
                )
            )
        )
    )

    kernel = kernels()
    loss = losses()
    ncf = non_conformity()

    predictor = kernel_regression(
        kernel.maker(param["predictor"]["kernel"]["name"]),
        param["predictor"]["kernel"]["params"],
        loss.maker(param["predictor"]["loss"]["name"]),
        param["predictor"]["loss"]["params"],
        param["predictor"]["lam"]["value"],
        param["predictor"]["lam"]["rate"],
    )

    cp_methods = {
        name: methods[name](
            predictor,
            ncf.maker(param["non_conformity"]["name"]),
            param["non_conformity"]["params"],
        )
        for name in param["methods"]
    }

    result = {"nb_rep": param["nb_rep"], "grid_n_samples": grid_n_samples, "data": []}

    for i_rep in range(param["nb_rep"]):
        gain = one_rep(i_rep, grid_n_samples, param, cp_methods)

        result["data"] += [
            {name: pd.DataFrame.from_dict(gain[name]) for name in param["methods"]}
        ]

    path_result = path_output + "result/"
    for name in param["methods"]:
        path_result += name

    path_result += (
        "/"
        + param["dataset"]
        + "_"
        + param["predictor"]["kernel"]["name"]
        + "_"
        + param["predictor"]["loss"]["name"]
        + "_"
        + "{}".format(param["predictor"]["lam"]["rate"])
        + ".npy"
    )

    save_result(result, path_result)
    return


def estimate_rate(x, y):
    mask = np.logical_not(np.logical_or(x == 0, y == 0))
    return (
        (np.log(x[mask]) * np.log(y[mask])).mean()
        - np.log(x[mask]).mean() * np.log(y[mask]).mean()
    ) / ((np.log(x[mask]) ** 2).mean() - (np.log(x[mask]).mean()) ** 2)


def name_maker(name, col, rate):
    res = r""
    if col == "confidence_gap":
        res += r"Implicit"
    elif col == "upper_bound":
        res += r"Explicit"
    res += r", Estimated slope ${:.2f}$".format(rate)
    return res


def display(path_param, path_output):
    param = param_reader(path_param)
    path_result = path_output + "result/"
    for name in param["methods"]:
        path_result += name

    path_result += (
        "/"
        + param["dataset"]
        + "_"
        + param["predictor"]["kernel"]["name"]
        + "_"
        + param["predictor"]["loss"]["name"]
        + "_"
        + "{}".format(param["predictor"]["lam"]["rate"])
        + ".npy"
    )

    result = result_reader(path_result)

    for i_rep in range(result["nb_rep"]):
        # theoretical upper bounds
        fig, ax = plt.subplots(figsize=(10, 8))

        for name in param["methods"]:
            for col in param["columns"]:
                rate = estimate_rate(
                    result["grid_n_samples"], result["data"][i_rep][name][col]
                )
                ax.plot(
                    result["grid_n_samples"],
                    result["data"][i_rep][name][col],
                    linestyle=param["visu"][name][col]["linestyle"],
                    linewidth=param["visu"][name][col]["linewidth"],
                    color=param["visu"][name][col]["color"],
                    marker=param["visu"][name][col]["marker"],
                    label=name_maker(name, col, rate),
                )

        ax.set_xlabel(r"Sample size $n$")
        ax.set_ylabel(r"Upper bound on the Thickness")

        title = "Evolution of upper bounds on the Thickness"
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(5e-3, 5e-1)
        ax.grid("on")
        ax.legend()

        path_display = path_output + "display/"
        for name in param["methods"]:
            path_display += name
        path_display += (
            "/"
            + param["dataset"]
            + "_"
            + param["predictor"]["kernel"]["name"]
            + "_"
            + param["predictor"]["loss"]["name"]
            + "_"
            + "{}".format(param["predictor"]["lam"]["rate"])
            + "_no{}".format(i_rep)
        )

        _ = [
            fig.savefig(
                path_display + "_ecas_sfds" + "." + form,
                format=form,
                bbox_inches="tight",
            )
            for form in ["jpeg", "eps"]
        ]
    return


# %%
path_params = [
    # "param/approx_fcp_0a.json",
    # "param/approx_fcp_1a.json",
    "param/approx_fcp_2a.json",
    # "param/approx_fcp_0b.json",
    # "param/approx_fcp_1b.json",
    # "param/approx_fcp_2b.json",
    # "param/approx_fcp_0c.json",
    # "param/approx_fcp_1c.json",
    # "param/approx_fcp_2c.json",
    # "param/approx_fcp_0d.json",
    # "param/approx_fcp_1d.json",
    # "param/approx_fcp_2d.json",
]
path_output = "output/"

# %%
# Here to start to run
# _ = [run(path_param, path_output) for path_param in path_params]

# %%
# Here to display the results only after running
_ = [display(path_param, path_output) for path_param in path_params]

# %%
