"""
Predictive models and fitting these models
"""
import numpy as np  # for the math
from sklearn.metrics.pairwise import pairwise_kernels
from scipy import optimize


def solve_kernel_regression(
    gram_matrix,
    output_points,
    lam=0.1,
    solver="lbfgs",
    max_iter=100,
    tol=1e-4,
):

    number_of_points = gram_matrix.shape[0]
    initial_model_weights = np.zeros(number_of_points, order="F")
    initial_model_weights = initial_model_weights.ravel(order="F")
    if solver not in ["lbfgs"]:
        raise ValueError("Only can handle this for now, sorry")
    else:
        empirical_risk = KernelLogisticEmpiricalRisk()
        empirical_risk_gradient = empirical_risk.empirical_risk_gradient
        optimization_result = optimize.minimize(
            empirical_risk_gradient,
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
    return final_model_weights


class KernelRegression:
    """
    Performs kernel regression
    """

    def __init__(
        self,
        lam=0.1,
        kernel="linear",
        gamma=None,
        degree=3.0,
        coef0=1.0,
        kernel_params=None,
        solver="lbfgs",
        max_iter=100
    ):
        self.name = "KernelRegression"
        self.lam = lam  # only one is supported

        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

        self.solver = solver
        self.max_iter = max_iter

        self.model_weights = None

        # I took this function from sklearn's KernelRidge
        def _get_kernel(self, X, Y=None):
            if callable(self.kernel):
                params = self.kernel_params or {}
            else:
                params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
            return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params)
        
        def fit(self, input_points, output_points):
            self.input_points = input_points
            gram_matrix = self._get_kernel(input_points)
            self.model_weights = solve_kernel_regression(
                gram_matrix,
                output_points,
                self.number_of_classes,
                self.lam,
                self.solver,
                self.max_iter,
            )
            return
