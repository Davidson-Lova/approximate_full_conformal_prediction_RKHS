"""
Predictive models and fitting these models
"""

import numpy as np  # for the math
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.optimize import _newton_cg
from scipy import optimize

from .kernel_empirical_risk import KernelEmpiricalRisk


def solve_kernel_regression(
    gram_matrix,
    output_points,
    lam=0.5,
    solver="newton-cg",
    max_iter=200,
    tol=1e-4,
    loss_name="log_cosh",
    loss_params={"alpha": 1.0},
):


    # initial_model_weights = np.zeros(output_points.shape)
    # initial_model_weights = initial_model_weights.ravel(order="F")

    initial_model_weights = gram_matrix @ np.random.normal(0, 1, output_points.shape)
    initial_model_weights = initial_model_weights.ravel(order="F")

    empirical_risk = KernelEmpiricalRisk(loss_name, loss_params)

    if solver not in ["lbfgs", "newton-cg"]:
        raise ValueError("Only can handle this for now, sorry")

    elif solver == "lbfgs":
        func = empirical_risk.empirical_risk_gradient
        optimization_result = optimize.minimize(
            func,
            initial_model_weights,
            method="L-BFGS-B",
            jac=True,
            args=(gram_matrix, output_points, lam),
            options={
                "maxiter": max_iter,
                "maxls": 50,  # default is 20
                "gtol": tol,
                "ftol": 64 * np.finfo(float).eps,
            },
        )
        final_model_weights = optimization_result.x

    elif solver == "newton-cg":
        func = empirical_risk.empirical_risk
        grad = empirical_risk.gradient
        hess = empirical_risk.gradient_hessian_product  # hess = [gradient, hessp]
        final_model_weights, _ = _newton_cg(
            grad_hess=hess,
            func=func,
            grad=grad,
            x0=initial_model_weights,
            args=(gram_matrix, output_points, lam),
            maxiter=max_iter,
            tol=tol,
        )
    return final_model_weights


class KernelRegression:
    """
    Performs kernel regression
    """

    def __init__(
        self,
        lam=0.5,
        kernel="linear",
        loss_name="log_cosh",
        loss_params={"alpha": 1.0},
        alpha=None,
        degree=3.0,
        coef0=1.0,
        kernel_params=None,
        solver="newton-cg",
        max_iter=200,
        tol=1e-4,
    ):
        self.name = "KernelRegression"
        self.lam = lam  # only one is supported

        self.kernel = kernel
        self.loss_name = loss_name
        self.loss_params = loss_params
        self.alpha = alpha
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol

        self.model_weights = None

    # I took this function from sklearn's KernelRidge
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {
                "alpha": self.alpha,
                "degree": self.degree,
                "coef0": self.coef0,
            }
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)

    def fit(self, input_points, output_points):
        self.input_points = input_points
        gram_matrix = self._get_kernel(input_points)
        self.model_weights = solve_kernel_regression(
            gram_matrix,
            output_points,
            self.lam,
            self.solver,
            self.max_iter,
            self.tol,
            self.loss_name,
            self.loss_params,
        )
        return

    def predict(self, new_input_points):
        if self.model_weights is None:
            raise ValueError("Please fit the model first.")

        prediction_feature_matrix = self._get_kernel(
            new_input_points, self.input_points
        )

        model_weights_is_flat = self.model_weights.ndim == 1
        if model_weights_is_flat:
            model_weights_size = self.model_weights.size
            number_of_points = self.input_points.shape[0]
            number_of_outputs = int(model_weights_size / number_of_points)
            model_weights = self.model_weights.reshape(
                (number_of_points, number_of_outputs), order="F"
            )

        predictions = prediction_feature_matrix @ model_weights
        return predictions
