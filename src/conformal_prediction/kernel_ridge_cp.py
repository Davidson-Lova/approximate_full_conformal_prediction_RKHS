"""The purpose of the present module is to build prediction sets using conformal prediction for kernel ridge regression.

It should be able to be used as follows:
```

train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)

predictor = Regression()

conformal_predictor = KernelRidgeConformalPredictor(predictor, non_conformity_name="absolute")
region_predictor = conformal_predictor.fit_predict(train_input_points, train_output_points, test_input_points)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
"""

import numpy as np
import portion as P
from ..models.losses import maker


# This piece of code is taken from Eugène Ndiaye's ridgeCP in stable_conformal_prediction
def kernel_ridge_region_predictor(A, B, confidence_control_level):
    n_samples = B.shape[0]
    negative_B = np.where(B < 0)[0]
    A[negative_B] *= -1
    B[negative_B] *= -1
    S, U, V = [], [], []

    for i in range(n_samples):

        if B[i] != B[-1]:
            tmp_u_i = (A[i] - A[-1]) / (B[-1] - B[i])
            tmp_v_i = -(A[i] + A[-1]) / (B[-1] + B[i])
            u_i, v_i = np.sort([tmp_u_i, tmp_v_i])
            U += [u_i]
            V += [v_i]

        elif B[i] != 0:
            tmp_uv = -0.5 * (A[i] + A[-1]) / B[i]
            U += [tmp_uv]
            V += [tmp_uv]

        if B[-1] > B[i]:
            S += [P.closed(U[i], V[i])]

        elif B[-1] < B[i]:
            intvl_u = P.openclosed(-np.inf, U[i])
            intvl_v = P.closedopen(V[i], np.inf)
            S += [intvl_u.union(intvl_v)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] < A[i]:
            S += [P.closedopen(U[i], np.inf)]

        elif B[-1] == B[i] and B[i] > 0 and A[-1] > A[i]:
            S += [P.openclosed(-np.inf, U[i])]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) <= abs(A[i]):
            S += [P.open(-np.inf, np.inf)]

        elif B[-1] == B[i] and B[i] == 0 and abs(A[-1]) > abs(A[i]):
            S += [P.empty()]

        elif B[-1] == B[i] and A[-1] == A[i]:
            S += [P.open(-np.inf, np.inf)]

        else:
            print("boom !!!")

    hat_y = np.sort([-np.inf] + U + V + [np.inf])
    size = hat_y.shape[0]
    prediction_region = P.empty()
    p_values = np.zeros(size)

    for i in range(size - 1):

        n_pvalue_i = 0.0
        intvl_i = P.closed(hat_y[i], hat_y[i + 1])

        for j in range(n_samples):
            n_pvalue_i += intvl_i in S[j]

        p_values[i] = n_pvalue_i / n_samples

        if p_values[i] > confidence_control_level:
            prediction_region = prediction_region.union(intvl_i)

    return prediction_region


def kernel_ridge_fit_predict(gram_matrix, lam, output_points):
    return np.linalg.solve(
        (gram_matrix / gram_matrix.shape[0]) + 2 * lam * np.eye(gram_matrix.shape[0]),
        (gram_matrix / gram_matrix.shape[0]) @ output_points,
    )


class KernelRidgeConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "kernel_ridge_cp"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]

    def fit_predict(self, train_input_points, train_output_points, test_input_points):
        """
        Prediction region function (as a function of the confidence level)
        for each test input point
        """

        def region_predictor(confidence_control_level):
            prediction_regions = []
            for test_input_point in test_input_points:
                augmented_input_points = np.concatenate(
                    (train_input_points, test_input_point.reshape(1, -1))
                )
                augmented_output_points = np.concatenate(
                    (train_output_points, np.zeros((1, train_output_points.shape[1])))
                )

                gram_matrix = self.predictor._get_kernel(augmented_input_points)
                A = (
                    augmented_output_points
                    - kernel_ridge_fit_predict(
                        gram_matrix, self.predictor.lam, augmented_output_points
                    )
                ).flatten()

                e_np1 = np.zeros(gram_matrix.shape[0])
                e_np1[-1] = 1
                B = (
                    e_np1
                    - kernel_ridge_fit_predict(
                        gram_matrix, self.predictor.lam, e_np1
                    ).flatten()
                )

                prediction_region = kernel_ridge_region_predictor(
                    A, B, confidence_control_level
                )
                prediction_regions.append(prediction_region)

            return prediction_regions

        return region_predictor
