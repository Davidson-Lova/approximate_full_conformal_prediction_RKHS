import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler


def load_data(dataset="diabetes"):
    """
    Taken from E. Ndiaye's Script
    """

    if dataset == "boston":
        # data_url = "http://lib.stat.cmu.edu/datasets/boston"
        # raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        raw_df = pd.read_csv("boston.csv", sep="\s+", skiprows=22, header=None)
        X_ = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        Y_ = raw_df.values[1::2, 2].reshape(-1, 1)
        # boston = datasets.load_boston()
        # X_ = boston.data
        # Y_ = boston.target

    if dataset == "diabetes":
        diabetes = datasets.load_diabetes()
        X_ = diabetes.data
        Y_ = diabetes.target.reshape(-1, 1)

    if dataset == "housingcalifornia":
        housing = datasets.fetch_california_housing()
        X_, Y_ = housing.data, housing.target.reshape(-1, 1)

    if dataset == "friedman1":
        X_, Y_ = datasets.make_friedman1(n_samples=500, n_features=100, noise=1)
        Y_ = Y_.reshape(-1, 1)

    if dataset == "synthetic":
        dense = 0.7
        n_samples, n_features = (500, 100)
        X_, Y_ = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            # random_state=random_state,
            n_informative=int(n_features * dense),
            noise=1,
        )
        Y_ = Y_.reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(X_)
    X_ = scaler.transform(X_)

    scaler_y = StandardScaler()
    scaler_y.fit(Y_)
    Y_ = scaler_y.transform(Y_)

    return X_, Y_
