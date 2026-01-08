"""
Predictive models and fitting these models
"""

import numpy as np  # for the math
import scipy as sp
from tqdm import tqdm

from .utils.utils import solveh_im


class kernel_regression:
    """Kernel regression"""

    def __init__(
        self,
        kernel,
        kernel_params,
        loss_maker,
        loss_params,
        lam,
        lam_rate=0.0,
    ):
        self.kernel = kernel
        self.kernel_params = kernel_params

        self.loss_maker = loss_maker
        self.loss_params = loss_params
        self.loss_bundle = self.loss_maker(**self.loss_params)

        self.lam = lam
        self.lam_rate = lam_rate

    def update_loss(self, loss_params):
        self.loss_bundle = self.loss_maker(**loss_params)

    def set_g_mat(self, X_eval, X_rep):
        self.K = self.kernel(X_eval, X_rep, **self.kernel_params)

    def newton(self, grad, hess, x0, max_iter=200, tol=1e-10, view_progress=False):
        """
        Newton method for minimizer search
        """
        x = np.array(x0, dtype=float)
        early_stop = False

        for i in (
            tqdm(range(max_iter), leave=False) if view_progress else range(max_iter)
        ):

            gradient = grad(x)
            hessian = hess(x)
            # delta_x = np.linalg.solve(hessian, -gradient.reshape(-1, 1)).flatten()
            delta_x = solveh_im(hessian, -gradient.reshape(-1, 1)).flatten()
            x = x + delta_x

            # print("x shape", x.shape)
            # print("dx : {:2f}".format(np.linalg.norm(delta_x)))

            if np.linalg.norm(delta_x) < tol:
                early_stop = True
                break

        if early_stop:
            print("early stop\n")
            print("dx : {:2f}".format(np.linalg.norm(delta_x)))
        else:
            print("last stop")
            print("dx : {:2f}".format(np.linalg.norm(delta_x)))
        return x

    # This we can leave first
    def predict(self, X_eval, X_rep, _a_=None):
        """
        Forward
        returns a flat vector
        """
        K_ = self.kernel(X_eval, X_rep, **self.kernel_params)
        if _a_ is None:
            # print("Hey, _a_ here", self._a_)
            y_pred = K_ @ self._a_
        else:
            y_pred = K_ @ _a_
        return y_pred

    # This stays
    def norm_K(self, X_rep, _a_):
        K_ = self.kernel(X_rep, X_rep, **self.kernel_params)
        return np.dot(_a_, np.dot(K_, _a_))

    def _risk(self, X_eval, X_rep, Y_eval):
        """
        empirical risk
        Y_eval must be flat
        returns a numpy scalar
        """
        N_eval = X_eval.shape[0]
        Y_eval = Y_eval.flatten()

        def risk(_a_):
            res = (
                np.sum(
                    self.loss_bundle["f"](Y_eval, self.predict(X_eval, X_rep, _a_)),
                    axis=0,
                )
                / N_eval
            )  # empirical risk part

            reg = self.norm_K(X_rep, _a_) * (
                self.lam / (N_eval**self.lam_rate)
            )  # regularization part

            # print("res", res)
            # print("reg", self.norm_K(X_rep, _a_) / N_eval)
            res += reg

            return res

        return risk

    def _grad(self, X_eval, X_rep, Y_eval):
        """
        Gradient of the empirical risk
        Y_eval must be flat
        return a flat numpy array
        """
        N_rep = X_rep.shape[0]
        N_eval = X_eval.shape[0]
        Y_eval = Y_eval.flatten()
        K_ = self.kernel(X_rep, X_rep, **self.kernel_params)

        def grad(_a_):

            diff = self.loss_bundle["df"](
                Y_eval, self.predict(X_eval, X_rep, _a_)
            )  # a flat vector
            reg_vec = (
                2
                * (
                    self.lam
                    / (
                        N_eval**self.lam_rate
                    )  # rate of decrease of the regularization parameter
                    # self.lam / N_eval
                )
                * (K_ @ _a_)
            )

            diff_er = np.sum((K_ * diff), axis=1) / N_eval
            # res = diff_er
            res = diff_er + reg_vec

            # # print("Hi, I'm grad", res.shape)
            return res

        return grad

    def _hessian(self, X_eval, X_rep, Y_eval):
        """
        Hessian of the empirical risk
        Y_eval must be flat
        returns a (X_rep.shape[0], X_rep.shape[0]) numpy array
        """
        N_eval = X_eval.shape[0]
        N_rep = X_rep.shape[0]
        Y_eval = Y_eval.flatten()

        K_eval_rep = self.kernel(X_eval, X_rep, **self.kernel_params)
        K_rep_rep = self.kernel(X_rep, X_rep, **self.kernel_params)

        def hess(_a_):

            diff2 = self.loss_bundle["ddf"](
                Y_eval, self.predict(X_eval, X_rep, _a_)
            )  # a flat vector

            reg_mat = (
                2
                * (
                    self.lam
                    / (
                        N_eval**self.lam_rate
                    )  # rate of decrease of the regularization parameter
                )
                * K_rep_rep
            )

            diff2_er = (
                K_eval_rep.transpose()
                @ (diff2 * K_eval_rep.transpose()).transpose()
                / N_eval
            )
            # res = diff2_er
            res = diff2_er + reg_mat

            # # print("Hi, I'm hess", res.shape)
            return res

        return hess

    def fit(
        self, X_eval, X_rep, Y_eval, _a_0=None, max_iter=100, tol=1e-6, custom=True
    ):
        """Optimal parameter search"""
        self.set_g_mat(X_eval, X_rep)

        if _a_0 is None:
            # Random initialization
            _a_0 = np.matmul(
                self.kernel(X_rep, X_rep, **self.kernel_params),
                np.random.normal(0, 1, (X_rep.shape[0], 1)),
            ).flatten()
            # print("_a_0", _a_0.shape)

        if custom:
            self._a_ = self.newton(
                self._grad(X_eval, X_rep, Y_eval),
                self._hessian(X_eval, X_rep, Y_eval),
                _a_0,
                max_iter,
                tol,
            )
        else:
            res = sp.optimize.minimize(
                self._risk(X_eval, X_rep, Y_eval),
                x0=_a_0,
                jac=self._grad(X_eval, X_rep, Y_eval),
                hess=self._hessian(X_eval, X_rep, Y_eval),
                method="Newton-CG",
            )
            self._a_ = res.x
            if res.success:
                print("Success")
                print(res.message)
            else:
                # print("Failure")
                print(res.message)
