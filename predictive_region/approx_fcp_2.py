"Corrected Approximate full conformal region"

import numpy as np

from .cp import cp
from .utils.utils import (
    abs_callable,
    add,
    comp,
    comp_2,
    min_loc,
    mul,
    p_value_maker,
    partial_2,
    region_maker,
    smallest_non_zero_eig,
    solveh_im,
    square_callable,
    sub,
)


class approx_fcp_2(cp):
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "approx_fcp_2"

    def _ncs_(self, X_train, Y_train, X_test, Z_test, params):

        N_test = X_test.shape[0]
        N_train_p1 = X_train.shape[0] + 1

        ncs = []
        K_diag = []
        hat_Y_Np1 = []
        d_loss_Np1_y = []
        d_loss_Np1_z = []
        K_diag_3half_mean = []
        d_ncs = []

        for j in range(N_test):
            # Parameter approximation
            X_aug = np.vstack((X_train, X_test[j, :]))
            Y_aug = np.vstack((Y_train, Z_test[j, :]))

            self.fit(X_aug, X_aug, Y_aug, params)

            hess = self.predictor._hessian(X_aug, X_aug, Y_aug)
            hat_H = hess(self.predictor._a_)

            # Compute K matrix
            K_diag += [np.diagonal(self.predictor.K)]
            K_diag_3half_mean += [np.mean(np.diagonal(self.predictor.K) ** 1.5)]

            #
            hat_Y_aug = self.predict(X_aug, X_aug)  # vec, flat
            hat_Y_Np1 += [hat_Y_aug[-1]]

            if_test_param_proto = (
                -solveh_im(hat_H, self.predictor.K[:, -1].reshape(-1, 1)) / N_train_p1
            )  # vec, mat

            if_test_pred_proto = np.matmul(self.predictor.K, if_test_param_proto)

            d_loss_Np1_y += [
                partial_2(self.predictor.loss_bundle["df"], hat_Y_aug[-1].item())
            ]

            d_loss_Np1_z += [d_loss_Np1_y[j](Z_test[j, :].item())]

            tilde_Y_y = [
                add(
                    mul(d_loss_Np1_y[j], if_test_pred_proto[i, :].item()),
                    hat_Y_aug[i].item()
                    - d_loss_Np1_z[j] * if_test_pred_proto[i, :].item(),
                )
                for i in range(N_train_p1)
            ]

            # Score approximation
            ncs += [
                [
                    comp(
                        self.non_conformity_bundle["f"],
                        tilde_Y_y[i],
                        Y_train[i, :].item(),
                    )
                    for i in range(N_train_p1 - 1)
                ]
            ]
            ncs[j] += [comp_2(self.non_conformity_bundle["f"], tilde_Y_y[-1])]

            if self.non_conformity_bundle["lams"]["reg"] != "C0":
                # Grad score function
                d_ncs += [
                    [
                        comp(
                            self.non_conformity_bundle["df"],
                            tilde_Y_y[i],
                            Y_train[i, :].item(),
                        )
                        for i in range(N_train_p1 - 1)
                    ]
                ]  # l(l(flat <- func))

                d_ncs[j] += [comp_2(self.non_conformity_bundle["df"], tilde_Y_y[-1])]

        return (
            ncs,
            K_diag,
            hat_Y_Np1,
            d_loss_Np1_y,
            d_loss_Np1_z,
            d_ncs,
            K_diag_3half_mean,
        )

    def predictor_qlty_bound(
        self, K_diag, K_diag_3half_mean, d_loss_Np1_y, d_loss_Np1_z
    ):

        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        rho_1 = [
            mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 0.5)
            for j in range(N_test)
        ]

        tilde_rho_1 = [
            mul(
                rho_1[j],
                1
                + (
                    K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"]
                    / (
                        (
                            self.predictor.lam
                            * (N_train_p1) ** (-self.predictor.lam_rate)
                        )
                        * N_train_p1
                    )
                ),
            )
            for j in range(N_test)
        ]

        rho_2 = [
            add(
                mul(
                    square_callable(tilde_rho_1[j]),
                    self.predictor.loss_bundle["lams"]["xi"]
                    * 0.5
                    * (K_diag[j][-1] ** 0.5)
                    * K_diag_3half_mean[j],
                ),
                mul(
                    tilde_rho_1[j],
                    2
                    * (self.predictor.lam * (N_train_p1) ** (-self.predictor.lam_rate))
                    * K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"],
                ),
            )
            for j in range(N_test)
        ]

        e_ = [
            min_loc(
                mul(
                    rho_2[j],
                    (K_diag[j][-1] ** 0.5)
                    / (
                        (
                            (
                                self.predictor.lam
                                * (N_train_p1) ** (-self.predictor.lam_rate)
                            )
                            ** 3
                        )
                        * (N_train_p1**2)
                    ),
                ),
                mul(
                    rho_1[j],
                    2
                    * (K_diag[j][-1] ** 0.5)
                    / (
                        (
                            self.predictor.lam
                            * (N_train_p1) ** (-self.predictor.lam_rate)
                        )
                        * N_train_p1
                    ),
                ),
            )
            for j in range(N_test)
        ]
        return e_

    def monitor_ncs_bound(
        self,
        K_diag,
        K_diag_3half_mean,
        d_loss_Np1_y,
        d_loss_Np1_z,
        d_ncs,
    ):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        rho_1 = [
            mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 0.5)
            for j in range(N_test)
        ]

        tilde_rho_1 = [
            mul(
                rho_1[j],
                1
                + (
                    K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"]
                    / (
                        (
                            self.predictor.lam
                            * (N_train_p1) ** (-self.predictor.lam_rate)
                        )
                        * N_train_p1
                    )
                ),
            )
            for j in range(N_test)
        ]

        rho_2 = [
            add(
                mul(
                    square_callable(tilde_rho_1[j]),
                    self.predictor.loss_bundle["lams"]["xi"]
                    * 0.5
                    * (K_diag[j][-1] ** 0.5)
                    * K_diag_3half_mean[j],
                ),
                mul(
                    tilde_rho_1[j],
                    2
                    * (self.predictor.lam * (N_train_p1) ** (-self.predictor.lam_rate))
                    * K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"],
                ),
            )
            for j in range(N_test)
        ]

        if self.non_conformity_bundle["lams"]["reg"] == "C0":
            gamma = [
                [self.non_conformity_bundle["lams"]["rho"] for i in range(N_train_p1)]
                for j in range(N_test)
            ]
        else:
            gamma = [
                [
                    add(
                        abs_callable(d_ncs[j][i]),
                        mul(
                            min_loc(
                                mul(
                                    rho_2[j],
                                    1
                                    / (
                                        (
                                            (
                                                self.predictor.lam
                                                * (N_train_p1)
                                                ** (-self.predictor.lam_rate)
                                            )
                                            ** 3
                                        )
                                        * (N_train_p1**2)
                                    ),
                                ),
                                mul(
                                    rho_1[j],
                                    2
                                    / (
                                        (
                                            self.predictor.lam
                                            * (N_train_p1) ** (-self.predictor.lam_rate)
                                        )
                                        * N_train_p1
                                    ),
                                ),
                            ),
                            (K_diag[j][i] ** 0.5)
                            * (K_diag[j][-1] ** 0.5)
                            * self.predictor.loss_bundle["lams"]["beta"],
                        ),
                    )
                    for i in range(N_train_p1)
                ]
                for j in range(N_test)
            ]

        return (
            tilde_rho_1,
            rho_2,
            gamma,
            self.predictor.loss_bundle["lams"]["beta"],
            self.predictor.loss_bundle["lams"]["xi"],
        )

    def ncs_qlty_bound(
        self,
        K_diag,
        d_loss_Np1_y,
        d_loss_Np1_z,
        d_ncs,
        K_diag_3half_mean,
    ):
        N_test = len(K_diag)
        N_train_p1 = K_diag[0].shape[0]

        rho_1 = [
            mul(abs_callable(add(d_loss_Np1_y[j], -d_loss_Np1_z[j])), 0.5)
            for j in range(N_test)
        ]

        tilde_rho_1 = [
            mul(
                rho_1[j],
                1
                + (
                    K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"]
                    / (
                        self.predictor.lam
                        * (N_train_p1) ** (1 - self.predictor.lam_rate)
                    )
                ),
            )
            for j in range(N_test)
        ]

        rho_2 = [
            add(
                mul(
                    square_callable(tilde_rho_1[j]),
                    0.5
                    * (K_diag[j][-1] ** 0.5)
                    * K_diag_3half_mean[j]
                    * self.predictor.loss_bundle["lams"]["xi"],
                ),
                mul(
                    tilde_rho_1[j],
                    2
                    * (self.predictor.lam * (N_train_p1) ** (-self.predictor.lam_rate))
                    * K_diag[j][-1]
                    * self.predictor.loss_bundle["lams"]["beta"],
                ),
            )
            for j in range(N_test)
        ]

        if self.non_conformity_bundle["lams"]["reg"] == "C0":
            tau = [
                [
                    mul(
                        min_loc(
                            mul(
                                rho_2[j],
                                1
                                / (
                                    (self.predictor.lam**3)
                                    * (
                                        (N_train_p1)
                                        ** (2 - 3 * self.predictor.lam_rate)
                                    )
                                ),
                            ),
                            mul(
                                rho_2[j],
                                2
                                / (
                                    self.predictor.lam
                                    * (N_train_p1) ** (1 - self.predictor.lam_rate)
                                ),
                            ),
                        ),
                        (K_diag[j][i] ** 0.5)
                        * (K_diag[j][-1] ** 0.5)
                        * self.non_conformity_bundle["lams"]["rho"],
                    )
                    for i in range(N_train_p1)
                ]
                for j in range(N_test)
            ]
        else:
            gamma = [
                [
                    add(
                        abs_callable(d_ncs[j][i]),
                        mul(
                            min_loc(
                                mul(
                                    rho_2[j],
                                    1
                                    / (
                                        (self.predictor.lam**3)
                                        * (
                                            (N_train_p1)
                                            ** (2 - 3 * self.predictor.lam_rate)
                                        )
                                    ),
                                ),
                                mul(
                                    rho_1[j],
                                    2
                                    / (
                                        self.predictor.lam
                                        * (N_train_p1) ** (1 - self.predictor.lam_rate)
                                    ),
                                ),
                            ),
                            (K_diag[j][i] ** 0.5)
                            * (K_diag[j][-1] ** 0.5)
                            * self.predictor.loss_bundle["lams"]["beta"],
                        ),
                    )
                    for i in range(N_train_p1)
                ]
                for j in range(N_test)
            ]
            tau = [
                [
                    mul(
                        gamma[j][i],
                        mul(
                            min_loc(
                                mul(
                                    rho_2[j],
                                    1
                                    / (
                                        (self.predictor.lam**3)
                                        * (
                                            (N_train_p1)
                                            ** (2 - 3 * self.predictor.lam_rate)
                                        )
                                    ),
                                ),
                                mul(
                                    rho_1[j],
                                    2
                                    / (
                                        self.predictor.lam
                                        * (N_train_p1) ** (1 - self.predictor.lam_rate)
                                    ),
                                ),
                            ),
                            (K_diag[j][i] ** 0.5) * (K_diag[j][-1] ** 0.5),
                        ),
                    )
                    for i in range(N_train_p1)
                ]
                for j in range(N_test)
            ]

        return tau

    def thickness_bound_explicit(self, K_diag):

        N_train_p1 = K_diag[0].shape[0]

        diag_maxes = [diag.max() for diag in K_diag]
        # print([type(diag) for diag in diag_maxes])
        facts = [
            1
            + (
                (diag_max * self.predictor.loss_bundle["lams"]["beta"])
                / ((self.predictor.lam * (N_train_p1) ** (1 - self.predictor.lam_rate)))
            )
            for diag_max in diag_maxes
        ]

        ms = [
            0.5
            * (diag_max**2)
            * (fact**2)
            * (self.predictor.loss_bundle["lams"]["rho"] ** 2)
            * self.predictor.loss_bundle["lams"]["xi"]
            + (
                2
                * (self.predictor.lam * (N_train_p1) ** (-self.predictor.lam_rate))
                * diag_max
                * self.predictor.loss_bundle["lams"]["beta"]
                * fact
                * self.predictor.loss_bundle["lams"]["rho"]
            )
            for diag_max, fact in zip(diag_maxes, facts)
        ]

        bounds = [
            (
                12
                * diag_max
                / (
                    1
                    - (
                        (diag_max * self.predictor.loss_bundle["lams"]["beta"])
                        / (
                            self.predictor.lam
                            * (N_train_p1) ** (1 - self.predictor.lam_rate)
                        )
                    )
                )
            )
            * min(
                m
                / (
                    (self.predictor.lam**3)
                    * (N_train_p1) ** (2 - 3 * self.predictor.lam_rate)
                ),
                (2 * self.predictor.loss_bundle["lams"]["rho"])
                / (self.predictor.lam * (N_train_p1) ** (1 - self.predictor.lam_rate)),
            )
            for diag_max, m in zip(diag_maxes, ms)
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
            K_diag_3half_mean,
        ) = self._ncs_(X_train, Y_train, X_test, Z_test, params)

        tau = self.ncs_qlty_bound(
            K_diag, d_loss_Np1_y, d_loss_Np1_z, d_ncs, K_diag_3half_mean
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
