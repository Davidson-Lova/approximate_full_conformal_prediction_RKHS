"""
The present aim is to illustrate the rate of decay of
+ the approximation error bounds
+ the p-value gap
+ the confidence gap
as a function of the training data set size under a grid of specification on
+ the data set
    + make friedman
    + make regression
+ the methods
    + approx_fcp smooth
    + approx_fcp very smooth
+ the predictor
    + kernel
    + loss function
    + base value of regularization parameter
    + rate of decay of regularization parameter
+ the non-conformity function.
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


def one_rep(i_rep, grid_n_samples, param, cp_methods):
    (
        mean_tau_1,
        max_tau_1,
        mean_p_value_gap_1,
        max_p_value_gap_1,
        confidence_region_gap_1,
        mean_tau_2,
        max_tau_2,
        mean_p_value_gap_2,
        max_p_value_gap_2,
        confidence_region_gap_2,
    ) = ([], [], [], [], [], [], [], [], [], [])

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

        y_min = np.min(y_seen)
        y_max = np.max(y_seen)
        grid_y = np.linspace(y_min, y_max, 5)

        # formulate bounds, conformal p-value function and region

        (
            ncs_1,
            K_diag_1,
            hat_Y_Np1_1,
            mus_1,
            d_loss_Np1_y_1,
            d_loss_Np1_z_1,
            d_ncs_1,
        ) = cp_methods["approx_fcp_1"]._ncs_(
            X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
        )

        tau_1 = cp_methods["approx_fcp_1"].ncs_qlty_bound(
            K_diag_1, mus_1, d_loss_Np1_y_1, d_loss_Np1_z_1, d_ncs_1
        )

        corrected_ncs_up_1 = cp_methods["approx_fcp_1"].corr_ncs_up(ncs_1, tau_1)
        p_value_function_up_1 = p_value_maker(corrected_ncs_up_1)
        region_up_1 = region_maker(p_value_function_up_1, y_min, y_max, hat_Y_Np1_1)

        corrected_ncs_lo_1 = cp_methods["approx_fcp_1"].corr_ncs_lo(ncs_1, tau_1)
        p_value_function_lo_1 = p_value_maker(corrected_ncs_lo_1)
        region_lo_1 = region_maker(p_value_function_lo_1, y_min, y_max, hat_Y_Np1_1)

        (
            ncs_2,
            K_diag_2,
            hat_Y_Np1_2,
            mus_2,
            d_loss_Np1_y_2,
            d_loss_Np1_z_2,
            d_ncs_2,
            K_diag_3half_mean_2,
        ) = cp_methods["approx_fcp_2"]._ncs_(
            X_train, y_seen, X_test, z_test, param["predictor"]["params_fit"]
        )

        tau_2 = cp_methods["approx_fcp_2"].ncs_qlty_bound(
            K_diag_2,
            mus_2,
            d_loss_Np1_y_2,
            d_loss_Np1_z_2,
            d_ncs_2,
            K_diag_3half_mean_2,
        )

        corrected_ncs_up_2 = cp_methods["approx_fcp_2"].corr_ncs_up(ncs_2, tau_2)
        p_value_function_up_2 = p_value_maker(corrected_ncs_up_2)
        region_up_2 = region_maker(p_value_function_up_2, y_min, y_max, hat_Y_Np1_2)

        corrected_ncs_lo_2 = cp_methods["approx_fcp_2"].corr_ncs_lo(ncs_2, tau_2)
        p_value_function_lo_2 = p_value_maker(corrected_ncs_lo_2)
        region_lo_2 = region_maker(p_value_function_lo_2, y_min, y_max, hat_Y_Np1_2)

        # eval bounds, p-value gap and region gap
        value_tau_1 = np.array([tau_1_i(grid_y) for tau_1_i in tau_1[0]])
        mean_tau_1 += [value_tau_1.mean()]
        max_tau_1 += [value_tau_1.max()]

        value_p_value_gap_1 = p_value_function_up_1[0](grid_y) - p_value_function_lo_1[
            0
        ](grid_y)
        mean_p_value_gap_1 += [value_p_value_gap_1.mean()]
        max_p_value_gap_1 += [value_p_value_gap_1.max()]

        value_region_up_1 = region_up_1(param["region"]["control_level"])
        value_region_lo_1 = region_lo_1(param["region"]["control_level"])
        confidence_region_gap_1 += [
            interval_length(value_region_up_1[0])
            - interval_length(value_region_lo_1[0])
        ]

        value_tau_2 = np.array([tau_2_i(grid_y) for tau_2_i in tau_2[0]])
        mean_tau_2 += [value_tau_2.mean()]
        max_tau_2 += [value_tau_2.max()]

        value_p_value_gap_2 = p_value_function_up_2[0](grid_y) - p_value_function_lo_2[
            0
        ](grid_y)
        mean_p_value_gap_2 += [value_p_value_gap_2.mean()]
        max_p_value_gap_2 += [value_p_value_gap_2.max()]

        value_region_up_2 = region_up_2(param["region"]["control_level"])
        value_region_lo_2 = region_lo_2(param["region"]["control_level"])
        confidence_region_gap_2 += [
            interval_length(value_region_up_2[0])
            - interval_length(value_region_lo_2[0])
        ]

    print("\n")

    return {
        "approx_fcp_1": [
            mean_tau_1,
            max_tau_1,
            mean_p_value_gap_1,
            max_p_value_gap_1,
            confidence_region_gap_1,
        ],
        "approx_fcp_2": [
            mean_tau_2,
            max_tau_2,
            mean_p_value_gap_2,
            max_p_value_gap_2,
            confidence_region_gap_2,
        ],
    }


def save_result(result, path):
    np.save(path, result)


def result_reader(path):
    return np.load(path, allow_pickle=True).tolist()


def run(param_path, result_path):
    param = param_reader(param_path)
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
        "approx_fcp_1": approx_fcp_1(
            predictor,
            ncf.maker(param["non_conformity"]["name"]),
            param["non_conformity"]["params"],
        ),
        "approx_fcp_2": approx_fcp_2(
            predictor,
            ncf.maker(param["non_conformity"]["name"]),
            param["non_conformity"]["params"],
        ),
    }

    columns = [
        "mean_tau",
        "max_tau",
        "mean_p_value_gap",
        "max_p_value_gap",
        "confidence_region_gap",
    ]

    result = {"nb_rep": param["nb_rep"], "grid_n_samples": grid_n_samples, "data": []}
    for i_rep in range(param["nb_rep"]):
        gain = one_rep(i_rep, grid_n_samples, param, cp_methods)
        result["data"] += [
            {
                "approx_fcp_1": pd.DataFrame(
                    np.array(gain["approx_fcp_1"]).transpose(), columns=columns
                ),
                "approx_fcp_2": pd.DataFrame(
                    np.array(gain["approx_fcp_2"]).transpose(), columns=columns
                ),
            }
        ]

    save_result(
        result,
        result_path
        + param["dataset"]
        + "_"
        + param["predictor"]["kernel"]["name"]
        + "_"
        + param["predictor"]["loss"]["name"]
        + ".npy",
    )
    return


def ax_labeler(ax, col):
    ax.set_xlabel(r"Sample size $n$")
    if col == "mean_tau" or col == "max_tau":
        ax.set_ylabel(r"Upper bounds $\tau$")
    elif col == "mean_p_value_gap" or col == "max_p_value_gap":
        ax.set_ylabel(r"p-value gap $\mathrm{Gap}\hat{\pi}_{D}(X_{n+1}, \cdot)$")
    elif col == "confidence_region_gap":
        ax.set_ylabel(r"Confidence region gap $\mathrm{Gap}\hat{C}_{\alpha}(X_{n+1})$")


def ax_titler(ax, col):
    if col == "mean_tau":
        ax.set_title(r"Evolution of the mean of the upper bounds")
    elif col == "max_tau":
        ax.set_title(r"Evolution of the maximum of the upper bounds")
    elif col == "mean_p_value_gap":
        ax.set_title(r"Evolution of the mean of the p-value gap")
    elif col == "max_p_value_gap":
        ax.set_title(r"Evolution of the maximum of the p-value gap")
    elif col == "confidence_region_gap":
        ax.set_title(r"Evolution of the confidence region gap")


def rate(x, y):
    mask = np.logical_not(np.logical_or(x == 0, y == 0))
    return np.cov(np.log(x[mask]), np.log(y[mask]))[0][1] / np.var(np.log(x[mask]))


def display(param_path, result_path, display_path):
    param = param_reader(param_path)
    result = result_reader(
        result_path
        + param["dataset"]
        + "_"
        + param["predictor"]["kernel"]["name"]
        + "_"
        + param["predictor"]["loss"]["name"]
        + ".npy"
    )

    columns = [
        "mean_tau",
        "max_tau",
        "mean_p_value_gap",
        "max_p_value_gap",
        "confidence_region_gap",
    ]

    for i_rep in range(result["nb_rep"]):
        for col in columns:
            rate_1 = rate(
                result["grid_n_samples"], result["data"][i_rep]["approx_fcp_1"][col]
            )
            rate_2 = rate(
                result["grid_n_samples"], result["data"][i_rep]["approx_fcp_2"][col]
            )

            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                result["grid_n_samples"],
                result["data"][i_rep]["approx_fcp_1"][col],
                "--bo",
                label="approx_fcp_1: {:.2f}".format(rate_1),
            )
            ax.plot(
                result["grid_n_samples"],
                result["data"][i_rep]["approx_fcp_2"][col],
                "--r+",
                label="approx_fcp_2: {:.2f}".format(rate_2),
            )

            ax_labeler(ax, col)
            ax_titler(ax, col)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim()
            ax.grid("on")
            ax.legend()

            _ = [
                fig.savefig(
                    display_path
                    + param["dataset"]
                    + "_"
                    + param["predictor"]["kernel"]["name"]
                    + "_"
                    + param["predictor"]["loss"]["name"]
                    + "_no{}".format(i_rep)
                    + "_"
                    + col
                    + "."
                    + form,
                    format=form,
                    bbox_inches="tight",
                )
                for form in ["jpeg", "eps"]
            ]
    return


# %%
param_path = "param/confidence_gap.json"
result_path = "result/"
display_path = "display/"

# %%
run(param_path, result_path)

# %%
display(param_path, result_path, display_path)
# %%
