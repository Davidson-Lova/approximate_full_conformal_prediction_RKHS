import numpy as np

from .cp import cp
from .utils.utils import add, p_value_maker, partial_2, region_maker, sub


class approx_fcp_0(cp):
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "approx_fcp_0"

    def _ncs_(self, X_train, Y_train, X_test, Z_test, params):

        N_test = X_test.shape[0]

        ncs = []
        K_diag = []
        hat_Y_Np1 = []

        for j in range(N_test):
            # Parameter approximation
            X_aug = np.vstack((X_train, X_test[j, :]))
            Y_aug = np.vstack((Y_train, Z_test[j, :]))

            self.fit(X_aug, X_aug, Y_aug, params)

            # Compute K matrix
            K_diag += [np.diagonal(self.predictor.K)]

            #
            hat_Y_aug = self.predict(X_aug, X_aug)  # vec, flat
            hat_Y_Np1 += [hat_Y_aug[-1]]

            # Score approximation
            ncs += [
                list(self.non_conformity_bundle["f"](Y_train.flatten(), hat_Y_aug[:-1]))
            ]
            ncs[j] += [partial_2(self.non_conformity_bundle["f"], hat_Y_aug[-1].item())]

        return (ncs, K_diag, hat_Y_Np1)

    def predictor_qlty_bound(self, K_diag):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        e_ = [
            ((K_diag[j][-1] ** 0.5) * self.predictor.loss_bundle["lams"]["rho"])
            / (self.predictor.lam * (N_train_p1) ** (1 - self.predictor.lam_rate))
            for j in range(N_test)
        ]

        return e_

    def ncs_qlty_bound(self, K_diag):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        tau = [
            [
                (K_diag[j][i] ** 0.5)
                * (K_diag[j][-1] ** 0.5)
                * self.non_conformity_bundle["lams"]["rho"]
                * self.predictor.loss_bundle["lams"]["rho"]
                / (self.predictor.lam * (N_train_p1) ** (1 - self.predictor.lam_rate))
                for i in range(N_train_p1)
            ]
            for j in range(N_test)
        ]
        return tau

    def thickness_bound_explicit(self, K_diag):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        bounds = [
            8
            * K_diag[j].max()
            * self.predictor.loss_bundle["lams"]["rho"]
            / (self.predictor.lam * (N_train_p1) ** (1 - self.predictor.lam_rate))
            for j in range(N_test)
        ]
        return bounds

    def corr_ncs_up(
        self,
        ncs,
        ncs_qlty_bounds,
    ):
        N_train = len(ncs_qlty_bounds[0]) - 1
        return [
            [
                add(score[i], bound[i]) if i < N_train else sub(score[i], bound[i])
                for i in range(N_train + 1)
            ]
            for score, bound in zip(ncs, ncs_qlty_bounds)
        ]

    def corr_ncs_lo(
        self,
        ncs,
        ncs_qlty_bounds,
    ):
        N_train = len(ncs_qlty_bounds[0]) - 1
        return [
            [
                sub(score[i], bound[i]) if i < N_train else add(score[i], bound[i])
                for i in range(N_train + 1)
            ]
            for score, bound in zip(ncs, ncs_qlty_bounds)
        ]

    def region(self, X_train, Y_train, X_test, Z_test, params):

        # Compute the regularity coefficients
        y_max = np.max(Y_train.flatten()).item()
        y_min = np.min(Y_train.flatten()).item()

        (ncs, K_diag, hat_Y_Np1) = self._ncs_(X_train, Y_train, X_test, Z_test, params)

        tau = self.ncs_qlty_bound(K_diag)

        corrected_ncs_up = self.corr_ncs_up(ncs, tau)
        p_value_function_up = p_value_maker(corrected_ncs_up)
        region_up = region_maker(p_value_function_up, y_min, y_max, hat_Y_Np1)

        corrected_ncs_low = self.corr_ncs_lo(ncs, tau)
        p_value_function_low = p_value_maker(corrected_ncs_low)
        region_low = region_maker(p_value_function_low, y_min, y_max, hat_Y_Np1)

        return {
            "up": {
                "region": region_up,
                "p_value_function": p_value_function_up,
            },
            "low": {"region": region_low, "p_value_function": p_value_function_low},
        }
