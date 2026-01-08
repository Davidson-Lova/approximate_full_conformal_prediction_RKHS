"Oracle conformal region"

import matplotlib.pyplot as plt
import numpy as np
import portion as P
from scipy.optimize import minimize_scalar, root_scalar
from sklearn.model_selection import train_test_split

from .cp import cp


class scp(cp):
    def __init__(
        self, predictor, non_conformity_maker, non_conformity_params, proper_train_size
    ):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "scp"
        self.proper_train_size = proper_train_size

    def _ncs_(self, X_train, Y_train, X_test, Y_test, params):
        # Train a predictor over training data

        X_p_train, X_calib, Y_p_train, Y_calib = train_test_split(
            X_train, Y_train, train_size=self.proper_train_size
        )

        self.fit(X_p_train, X_p_train, Y_p_train, params)
        nc_ncs_calib = self.compute_ncs(X_calib, X_p_train, Y_calib).flatten()
        nc_ncs_test = self.compute_ncs(X_test, X_p_train, Y_test).flatten()

        res = [list(nc_ncs_calib) + [test_score] for test_score in nc_ncs_test]
        N_calib = X_calib.shape[0]
        return res, N_calib

    def pvalues(self, X_train, Y_train, X_test, Y_test, params):
        scores_, N_calib = self._ncs_(X_train, Y_train, X_test, Y_test, params)
        res = [(sum(np.array(s[:-1]) >= s[-1]) + 1) / (N_calib + 1) for s in scores_]
        return res

    def region(self, X_train, Y_train, X_test, params):
        y_max = np.max(Y_train.flatten()).item()
        y_min = np.min(Y_train.flatten()).item()

        X_p_train, X_calib, Y_p_train, Y_calib = train_test_split(
            X_train, Y_train, train_size=self.proper_train_size
        )

        self.fit(X_p_train, X_p_train, Y_p_train, params)
        nc_ncs_calib = self.compute_ncs(X_calib, X_p_train, Y_calib).flatten()
        Y_hat_test = self.predict(X_test, X_p_train).flatten()

        N_calib = X_calib.shape[0]

        sorted_cal_scores = np.sort(nc_ncs_calib)

        def region_(control_level):
            index = int(np.ceil((N_calib + 1) * (1 - control_level)))
            qHat = sorted_cal_scores[index]

            # qlevel = np.ceil((N_calib + 1) * (1 - control_level)) / N_calib
            # qHat = np.quantile(nc_ncs_calib, qlevel, method="higher")

            def func_maker(y_hat_j):
                def res(y):
                    return qHat - self.non_conformity_bundle["f"](y, y_hat_j)

                return res

            funcs = [func_maker(y_hat_j) for y_hat_j in Y_hat_test]
            zls = [
                (f(y_min) * f(y_hat_j)) >= 0 for f, y_hat_j in zip(funcs, Y_hat_test)
            ]
            zus = [
                (f(y_max) * f(y_hat_j)) >= 0 for f, y_hat_j in zip(funcs, Y_hat_test)
            ]

            def lower_bound(f, y_hat_j, take_border):
                if take_border:
                    return y_min
                else:
                    lb_search = root_scalar(f, bracket=[y_min, y_hat_j], rtol=1e-10)
                    lb = lb_search.root if lb_search.converged else y_min
                return lb

            def upper_bound(f, y_hat_j, take_border):
                if take_border:
                    return y_max
                else:
                    ub_search = root_scalar(f, bracket=[y_hat_j, y_max], rtol=1e-10)
                    ub = ub_search.root if ub_search.converged else y_max
                return ub

            predictive_region = [
                P.closed(lower_bound(f, y_hat_j, zl), upper_bound(f, y_hat_j, zu))
                for f, y_hat_j, zl, zu in zip(funcs, Y_hat_test, zls, zus)
            ]

            return predictive_region

        return {"region": region_}
