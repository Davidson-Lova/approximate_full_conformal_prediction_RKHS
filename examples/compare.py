"""
The aim of the present script is to compare
the performances (size, coverage, computation time)
of the following methods:
    + full conformal,
    + split conformal, (try difference splitting ratios)
    + stable conformal,
    + local stable conformal,
    + approximate full conformal via influence function (vanilla and corrected).

The predictor will be kernel ridge regression.

    For kernel
        + Laplacian (0.1),
        + Linear.
    For the loss function
        + Quadratic (to get the actual full conformal),
        + Smooth pinball.

The score function is set to be the absolute deviation

Data set
"""

# %%
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("..")
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
from predictive_region.utils.tools import load_data
from predictive_region.utils.utils import interval_length

methods = {
    "oracle_cp": oracle_cp,
    "scp": scp,
    "approx_fcp_0": approx_fcp_0,
    "approx_fcp_1": approx_fcp_1,
    "approx_fcp_2": approx_fcp_2,
}


# %%
def set_style():
    plt.rcParams["text.usetex"] = True
    font = {"family": "normal", "weight": "bold", "size": 22}
    plt.rc("font", **font)

    # # This sets reasonable defaults for font size for
    # # a figure that will go in a paper
    # sns.set_context("paper")
    # # Set the font to be serif, rather than sans
    # sns.set(font="serif", font_scale=0.75)
    # sns.set_palette("muted")
    # # Make the background white, and specify the
    # # specific font family
    # sns.set_style(
    #     "whitegrid",
    #     {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    # )


set_style()


def param_reader(param_path):
    with open(param_path, "r") as file:
        params = json.load(file)
    return params


def method_maker(
    method_name,
    method,
    predictor,
    non_conformity_maker,
    non_conformity_params,
    proper_train_size=0.5,
):
    if method_name == "scp":
        res = method(
            predictor,
            non_conformity_maker,
            non_conformity_params,
            proper_train_size,
        )
    else:
        res = method(
            predictor,
            non_conformity_maker,
            non_conformity_params,
        )
    return res


# Method loader
def conformal_region_maker(
    cp_method, X_train, X_test, y_seen, y_not_seen, z_, params_fit
):
    if (cp_method.name == "oracle_cp") or (cp_method.name == "fcp_krr"):
        predictive_region_maker = cp_method.region(
            X_train, y_seen, X_test, y_not_seen, params_fit
        )["region"]
    elif cp_method.name == "scp":
        predictive_region_maker = cp_method.region(
            X_train,
            y_seen,
            X_test,
            params_fit,
        )["region"]
    else:
        predictive_region_maker = cp_method.region(
            X_train, y_seen, X_test, z_, params_fit
        )["up"]["region"]

    return predictive_region_maker


def record(
    cp_method, X_train, X_test, y_seen, y_not_seen, z_, params_fit, control_level
):
    print(cp_method.name)
    tic = time.time()

    cp_set_maker = conformal_region_maker(
        cp_method, X_train, X_test, y_seen, y_not_seen, z_, params_fit
    )

    cp_set = cp_set_maker(control_level)[0]

    tac = time.time()
    duration = tac - tic

    return np.array(
        [float(y_not_seen in cp_set), interval_length(cp_set), duration], dtype=object
    )


def one_rep(result, i_rep, cp_methods, params):
    print(i_rep, sep=" ", end=" ", flush=True)

    # generate data
    X_, y_ = load_data(params["data"])
    random_int = np.arange(y_.shape[0])
    np.random.shuffle(random_int)
    X_, y_ = X_[random_int, :], y_[random_int, :]

    X_train, X_test = X_[:-1, :], X_[-1, :].reshape(1, -1)
    y_seen, y_not_seen = y_[:-1, :], y_[-1, :].reshape(1, -1)
    z_ = np.zeros(y_not_seen.shape)

    # normalization
    X_scalar = StandardScaler()
    X_scalar.fit(X_train)
    X_train = X_scalar.transform(X_train)
    X_test = X_scalar.transform(X_test)

    #
    y_scalar = StandardScaler()
    y_scalar.fit(y_seen)
    y_seen = y_scalar.transform(y_seen)
    y_not_seen = y_scalar.transform(y_not_seen)

    for key in result.keys():
        result[key].iloc[i_rep] = record(
            cp_methods[key],
            X_train,
            X_test,
            y_seen,
            y_not_seen,
            z_,
            params["predictor"]["params_fit"],
            params["region"]["control_level"],
        )


def one_set(nb_rep, params):
    kernel = kernels()
    loss = losses()
    ncf = non_conformity()

    predictor = kernel_regression(
        kernel.maker(params["predictor"]["kernel"]["name"]),
        params["predictor"]["kernel"]["params"],
        loss.maker(params["predictor"]["loss"]["name"]),
        params["predictor"]["loss"]["params"],
        params["predictor"]["lam"]["value"],
        params["predictor"]["lam"]["rate"],
    )

    # build methods
    cp_methods = {
        key: method_maker(
            key,
            methods[key],
            predictor,
            ncf.maker(params["non_conformity"]["name"]),
            params["non_conformity"]["params"],
            params["methods"]["scp"]["proper_train_size"],
        )
        for key in methods.keys()
    }

    columns = ["coverage", "length", "time"]

    result = {}
    for key in methods.keys():
        result[key] = pd.DataFrame(np.zeros((nb_rep, len(columns))), columns=columns)

    _ = [one_rep(result, i_rep, cp_methods, params) for i_rep in range(nb_rep)]
    return result


def save_result(result, path):
    np.save(path, result)


def result_reader(path):
    return np.load(path, allow_pickle=True).tolist()


def display_results(result, params, fig, ax):
    print("Plot:", params["data"])
    print(
        params["predictor"]["kernel"]["name"],
        "kernel and ",
        params["predictor"]["loss"]["name"],
        "loss function",
    )
    print(params["non_conformity"]["name"], "non conformity_function")
    print("Rate of decay ", params["predictor"]["lam"]["rate"])
    print(
        "control_level =",
        params["region"]["control_level"],
        "and n_repet =",
        params["nb_rep"],
    )

    def labelize(name, df, norm=1):
        len = r"$\overline{\mathrm{length}}$ = "
        mean_len = str(np.round(df["length"].mean(), 2))
        cov = r"$\overline{cov}$ = "
        # var_cov = r"$\pm$" + str(np.round(3 * df["coverage"].std(), 2))
        # mean_cov = str(df["coverage"].mean())
        mean_cov = str(np.round(df["coverage"].mean(), 2))
        Ts = r"$\overline{T}$ = "
        mean_time = str(np.round(df["time"].mean() / norm, 2))
        return (
            name + "\n" + len + mean_len + "\n" + cov + mean_cov + "\n" + Ts + mean_time
        )
        # return name + " \n" + cov + mean_cov

    labels = []
    df_length = []
    for key in params["methods"].keys():
        # for key in result.keys():
        # if key not in ["scp", "oracle_cp"]:
        print("\n ", key, "\n", result[key].mean(), "\n")
        labels += [labelize(key, result[key], result["oracle_cp"]["time"].mean())]
        df_length += [result[key]["length"]]

    box = ax.boxplot(df_length, patch_artist=True)
    ax.set_ylabel("Length")

    # keys = ["approx_fcp_0", "approx_fcp_1", "approx_fcp_2"]
    # for patch, method_name in zip(
    #     box['boxes'], keys
    # ):
    #     patch.set_facecolor(params["display"]["colors"][method_name])

    for patch, method_name in zip(box["boxes"], params["methods"].keys()):
        # for patch, method_name in zip(
        #     box['boxes'], result.keys()
        # ):
        patch.set_facecolor(params["display"]["colors"][method_name])

    ax.grid(True)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    fig.tight_layout()
    return


def save_display(fig, params, display_path):
    _ = [
        fig.savefig(
            display_path
            + "length_"
            + params["data"]
            + "_"
            + params["predictor"]["kernel"]["name"]
            + "_"
            + params["predictor"]["loss"]["name"]
            + "."
            + format,
            format=format,
        )
        for format in params["display"]["formats"]
    ]


def run_0(param_path, result_path):
    params = param_reader(param_path)
    result = one_set(params["nb_rep"], params)
    save_result(
        result,
        result_path
        + params["data"]
        + "_"
        + params["predictor"]["kernel"]["name"]
        + "_"
        + params["predictor"]["loss"]["name"]
        + ".npy",
    )
    return


def run_1(param_path, result_path, display_path):
    params = param_reader(param_path)
    result = result_reader(
        result_path
        + params["data"]
        + "_"
        + params["predictor"]["kernel"]["name"]
        + "_"
        + params["predictor"]["loss"]["name"]
        + ".npy"
    )
    fig, ax = plt.subplots(figsize=tuple(params["display"]["fig_size"]))
    display_results(result, params, fig, ax)
    save_display(fig, params, display_path)
    return


# %%
param_path = "params/params_2.json"
result_path = "results/"
display_path = "results/"

# this takes time so only run when necessary
# run_0(param_path, result_path)

# %%
# run_1(param_path, result_path, display_path)

# %%
