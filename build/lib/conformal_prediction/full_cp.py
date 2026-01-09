"Corrected Approximate full conformal region"

# import matplotlib.pyplot as plt
import numpy as np

from .base_cp import cp
from .utils.utils import p_value_maker, region_maker


#
def lin_square(a, b):
    def res(y):
        return (a + b * y) ** 2

    return res


#
def lin_abs(a, b):
    def res(y):
        return np.abs(a + b * y)

    return res


class fcp(cp):
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        super().__init__(predictor, non_conformity_maker, non_conformity_params)
        self.name = "fcp_krr"

    def _ncs_(self, X_train, Y_train, X_test, Z_test, params):

        N_train = X_train.shape[0]
        N_test = X_test.shape[0]
        Z_test = np.zeros(Z_test.shape)

        ncs = []
        hat_Y_Np1 = []

        for j in range(N_test):
            # Parameter approximation
            X_aug = np.vstack((X_train, X_test[j, :]))
            Y_aug = np.vstack((Y_train, Z_test[j, :]))

            # train
            K_ = self.predictor.kernel(X_aug, X_aug, **self.predictor.kernel_params)
            C_ = K_.copy()
            lam = self.predictor.lam * ((N_train + 1) ** -self.predictor.lam_rate)
            C_.flat[:: C_.shape[1] + 1] += lam * (N_train + 1)

            invC_Y = np.linalg.solve(C_, Y_aug)
            eNp1 = np.zeros(Y_aug.shape)
            eNp1[-1, :] = 1.0
            invC_eNp1 = np.linalg.solve(C_, eNp1)

            K_invC_Y = K_ @ invC_Y
            K_invC_eNp1 = K_ @ invC_eNp1

            a_ = Y_aug - K_invC_Y
            b_ = eNp1 - K_invC_eNp1

            ncs += [
                [
                    lin_square(a.item(), b.item())
                    for a, b in zip(a_.flatten(), b_.flatten())
                ]
            ]

            hat_Y_Np1 += [K_invC_Y.flatten()[-1].item()]

        return ncs, hat_Y_Np1

    def region(self, X_train, Y_train, X_test, Z_test, params):

        # Compute the regularity coefficients
        y_max = np.max(Y_train.flatten()).item()
        y_min = np.min(Y_train.flatten()).item()

        ncs, hat_Y_Np1 = self._ncs_(X_train, Y_train, X_test, Z_test, params)

        p_value_function = p_value_maker(ncs)
        region_ = region_maker(p_value_function, y_min, y_max, hat_Y_Np1)

        return {"p_value_function": p_value_function, "region": region_}
