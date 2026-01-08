"Oracle conformal region"

import numpy as np
import portion as P
from scipy.optimize import root_scalar

from .cp import cp


class oracle_cp(cp):
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "oracle_cp"

    def _ncs_(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        params,
    ):
        # Train a predictor over training data
        N_test = X_test.shape[0]

        res = []
        for j in range(N_test):

            X_aug = np.vstack([X_train, X_test[j, :]])
            Y_aug = np.vstack([Y_train, Y_test[j, :]])
            self.fit(X_aug, X_aug, Y_aug, params)
            nc_ncs_j = self.compute_ncs(X_aug, X_aug, Y_aug)

            res += [list(nc_ncs_j.flatten())]

        return res

    def pvalues(self, X_train, Y_train, X_test, Y_test, params):
        scores_ = self._ncs_(X_train, Y_train, X_test, Y_test, params)
        res = [
            (sum(np.array(s[:-1]) >= s[-1]) + 1) / (X_train.shape[0] + 1)
            for s in scores_
        ]
        return res

    def region(self, X_train, Y_train, X_test, Y_test, params):

        y_max = np.max(Y_train.flatten()).item()
        y_min = np.min(Y_train.flatten()).item()

        N_train = X_train.shape[0]
        N_test = X_test.shape[0]

        nc_ncs_j = []
        y_hat_j = []

        for j in range(N_test):

            X_aug = np.vstack([X_train, X_test[j, :]])
            Y_aug = np.vstack([Y_train, Y_test[j, :]])

            self.fit(X_aug, X_aug, Y_aug, params)

            nc_ncs_j += [self.compute_ncs(X_train, X_aug, Y_train)]
            y_hat_j += [self.predict(X_test[j, :].reshape(1, -1), X_aug).flatten()[0]]

        def region_(control_level):
            predictive_region = []

            for j in range(N_test):
                qlevel = np.ceil((N_train + 1) * (1 - control_level)) / N_train
                qHat = np.quantile(nc_ncs_j[j].flatten(), qlevel, method="higher")

                def dsnp1(y):
                    return qHat - self.non_conformity_bundle["f"](y, y_hat_j[j])

                if (dsnp1(y_min) * dsnp1(y_hat_j[j])) >= 0:
                    lb = y_min
                else:
                    lb_search = root_scalar(
                        dsnp1, bracket=[y_min, y_hat_j[j]], rtol=1e-10
                    )
                    lb = lb_search.root if lb_search.converged else y_min

                if (dsnp1(y_max) * dsnp1(y_hat_j[j])) >= 0:
                    ub = y_max
                else:
                    ub_search = root_scalar(
                        dsnp1, bracket=[y_hat_j[j], y_max], rtol=1e-10
                    )
                    ub = ub_search.root if ub_search.converged else y_max

                predictive_region += [P.closed(lb, ub)]

            return predictive_region

        return {"region": region_}
