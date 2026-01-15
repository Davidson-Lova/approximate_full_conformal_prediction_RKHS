"Corrected Approximate full conformal region"

# import matplotlib.pyplot as plt
import numpy as np

from .base_cp import cp
from .utils.utils import (
    abs_callable,
    add,
    mul,
    p_value_maker,
    partial_2,
    region_maker,
    smallest_non_zero_eig,
    sub,
)


class approx_fcp_1(cp):
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "approx_fcp_1"

    def _ncs_(self, X_train, Y_train, X_test, Z_test, params):

        N_test = X_test.shape[0]

        ncs = []
        K_diag = []
        hat_Y_Np1 = []
        d_loss_Np1_y = []
        d_loss_Np1_z = []
        d_ncs = []

        for j in range(N_test):
            # Parameter approximation
            X_aug = np.vstack((X_train, X_test[j, :]))
            Y_aug = np.vstack((Y_train, Z_test[j, :]))

            self.fit(X_aug, X_aug, Y_aug, params)

            # Compute K matric
            K_diag += [np.diagonal(self.predictor.K)]

            #
            hat_Y_aug = self.predict(X_aug, X_aug)  # vec, flat
            hat_Y_Np1 += [hat_Y_aug[-1]]

            # Score approximation
            ncs += [
                list(self.non_conformity_bundle["f"](Y_train.flatten(), hat_Y_aug[:-1]))
            ]
            ncs[j] += [partial_2(self.non_conformity_bundle["f"], hat_Y_aug[-1].item())]

            #
            d_loss_Np1_y += [
                partial_2(self.predictor.loss_bundle["df"], hat_Y_aug[-1].item())
            ]
            d_loss_Np1_z += [d_loss_Np1_y[j](Z_test[j, :].item())]

            if self.non_conformity_bundle["lams"]["reg"] != "C0":
                # Grad score function
                d_ncs += [
                    list(
                        self.non_conformity_bundle["df"](
                            Y_train.flatten(), hat_Y_aug[:-1].flatten()
                        )
                    )
                ]  # l(l(flat <- fusnc))

                d_ncs[j] += [
                    partial_2(self.non_conformity_bundle["df"], hat_Y_aug[-1].item())
                ]

        return (
            ncs,
            K_diag,
            hat_Y_Np1,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
        )

    # def predictor_qlty_bound(self, K_diag, d_loss_Np1_y, d_loss_Np1_z, d_ncs):
    #     N_test = len(K_diag)
    #     N_train_p1 = K_diag[0].shape[0]

    #     rho_1 = [
    #         mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 1 / 2)
    #         for j in range(N_test)
    #     ]

    #     e_ = [
    #         mul(
    #             mul(rho_1[j], K_diag[j][-1] ** 0.5),
    #             1 / (self.predictor.lam * (N_train_p1)**(1-self.predictor.lam_rate)),
    #         )
    #         for j in range(N_test)
    #     ]

    #     return e_

    # def monitor_ncs_bound(self, K_diag, d_loss_Np1_y, d_loss_Np1_z, d_ncs):
    #     N_test = len(K_diag)
    #     N_train_p1 = K_diag[0].shape[0]

    #     rho_1 = [
    #         mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 1 / 2)
    #         for j in range(N_test)
    #     ]

    #     if self.non_conformity_bundle["lams"]["reg"] == "C0":
    #         alpha = [
    #             [self.non_conformity_bundle["lams"]["rho"] for i in range(N_train_p1)]
    #             for j in range(N_test)
    #         ]
    #     else:
    #         _alpha = [
    #             [
    #                 add(
    #                     mul(
    #                         rho_1[j],
    #                         (
    #                             (K_diag[j][i] ** 0.5)
    #                             * (K_diag[j][-1] ** 0.5)
    #                             * self.non_conformity_bundle["lams"]["beta"]
    #                         )
    #                         / (self.predictor.lam * (N_train_p1)**(1-self.predictor.lam_rate)),
    #                     ),
    #                     np.abs(d_ncs[j][i]),
    #                 )
    #                 for i in range(N_train_p1 - 1)
    #             ]
    #             for j in range(N_test)
    #         ]

    #         alpha = [
    #             _alpha[j]
    #             + [
    #                 add(
    #                     mul(
    #                         rho_1[j],
    #                         (K_diag[j][-1] * self.non_conformity_bundle["lams"]["beta"])
    #                         / (self.predictor.lam * (N_train_p1)**(1-self.predictor.lam_rate)),
    #                     ),
    #                     abs_callable(d_ncs[j][-1]),
    #                 )
    #             ]
    #             for j in range(N_test)
    #         ]

    #     return (
    #         rho_1,
    #         self.predictor.loss_bundle["lams"]["eta"],
    #         self.predictor.lam * (N_train_p1**-self.predictor.lam_rate),
    #         [1 / (lam * N_train_p1) for lam in lams],
    #         [1 / ((lam**3) * (N_train_p1**2)) for lam in lams],
    #         alpha,
    #     )

    def ncs_qlty_bound_0(self, K_diag):
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

    def ncs_qlty_bound(self, K_diag, d_loss_Np1_y, d_loss_Np1_z, d_ncs):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        rho_1 = [
            mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 1 / 2)
            for j in range(N_test)
        ]

        if self.non_conformity_bundle["lams"]["reg"] == "C0":
            tau = [
                [
                    mul(
                        rho_1[j],
                        (
                            (K_diag[j][i] ** 0.5)
                            * (K_diag[j][-1] ** 0.5)
                            * self.non_conformity_bundle["lams"]["rho"]
                        )
                        / (
                            self.predictor.lam
                            * (N_train_p1) ** (1 - self.predictor.lam_rate)
                        ),
                    )
                    for i in range(N_train_p1)
                ]
                for j in range(N_test)
            ]
        else:
            _alpha = [
                [
                    add(
                        mul(
                            rho_1[j],
                            (
                                (K_diag[j][i] ** 0.5)
                                * (K_diag[j][-1] ** 0.5)
                                * self.non_conformity_bundle["lams"]["beta"]
                            )
                            / (
                                self.predictor.lam
                                * (N_train_p1) ** (1 - self.predictor.lam_rate)
                            ),
                        ),
                        np.abs(d_ncs[j][i]),
                    )
                    for i in range(N_train_p1 - 1)
                ]
                for j in range(N_test)
            ]

            alpha = [
                _alpha[j]
                + [
                    add(
                        mul(
                            rho_1[j],
                            (K_diag[j][-1] * self.non_conformity_bundle["lams"]["beta"])
                            / (
                                self.predictor.lam
                                * (N_train_p1) ** (1 - self.predictor.lam_rate)
                            ),
                        ),
                        abs_callable(d_ncs[j][-1]),
                    )
                ]
                for j in range(N_test)
            ]

            tau = [
                [
                    mul(
                        alpha[j][i],
                        mul(
                            rho_1[j],
                            ((K_diag[j][i] ** 0.5) * (K_diag[j][-1] ** 0.5))
                            / (
                                self.predictor.lam
                                * (N_train_p1) ** (1 - self.predictor.lam_rate)
                            ),
                        ),
                    )
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

        (
            ncs,
            K_diag,
            hat_Y_Np1,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
        ) = self._ncs_(X_train, Y_train, X_test, Z_test, params)

        tau = self.ncs_qlty_bound(
            K_diag,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
        )

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
