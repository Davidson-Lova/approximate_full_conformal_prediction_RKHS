"""The purpose of the present module is to build prediction sets using approximate conformal prediction via influence function
for kernel ridge regression.

It should be able to be used as follows:
```

train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)

predictor = Regression()

conformal_predictor = InfluenceFunctionConformalPredictor(predictor, non_conformity_name="absolute")
region_predictor = conformal_predictor.fit_predict(train_input_points, train_output_points, test_input_points)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
"""

import numpy as np
import matplotlib.pyplot as plt
from ..models.losses import maker
from ..models.kernel_empirical_risk import KernelEmpiricalRisk
from .utils import inter_finder


def compute_approximate_predictions_(gram_matrix, dloss, proto_influence, predictions):
    def approximate_predictions_(output_value):
        proto_influence_predictions = gram_matrix @ proto_influence
        influence_predictions_z = (
            (-1.0)
            * proto_influence_predictions
            * (
                dloss(np.zeros(predictions[-1, :].shape), predictions[-1, :])
                / gram_matrix.shape[0]
            )
        )
        influence_predictions_y = (
            (-1.0)
            * proto_influence_predictions
            * (dloss(output_value, predictions[-1, :]) / gram_matrix.shape[0])
        )
        return predictions - influence_predictions_z + influence_predictions_y

    return approximate_predictions_


def compute_train_scores_(
    non_conformity, train_output_points, approximate_predictions_
):
    def train_scores_(output_value):
        approximate_predictions = approximate_predictions_(output_value)
        train_scores = non_conformity(
            train_output_points,
            approximate_predictions[:-1, :].reshape(train_output_points.shape),
        ).flatten()
        return train_scores

    return train_scores_


def compute_loss_rho_(dloss, test_prediction):
    def loss_rho_(output_value):
        loss_rho = 0.5 * np.abs(
            dloss(output_value, test_prediction)
            - dloss(np.zeros(test_prediction.shape), test_prediction)
        )
        return loss_rho

    return loss_rho_


def compute_big_loss_rho_(gram_matrix, lam, loss_beta, loss_rho_):
    def big_loss_rho_(output_value):
        big_loss_rho = (
            1 + (gram_matrix[-1, -1] * loss_beta) / (lam * gram_matrix.shape[0])
        ) * loss_rho_(output_value)
        return big_loss_rho

    return big_loss_rho_


def compute_bigger_loss_rho_(gram_matrix, lam, loss_beta, big_loss_rho_, loss_xi):
    def bigger_loss_rho_(output_value):
        bigger_loss_rho = 0.5 * loss_xi * np.sqrt(gram_matrix[-1, -1]) * np.mean(
            np.diag(gram_matrix) ** (1.5)
        ) * (big_loss_rho_(output_value) ** 2) + 2 * lam * gram_matrix[
            -1, -1
        ] * loss_beta * big_loss_rho_(
            output_value
        )
        return bigger_loss_rho

    return bigger_loss_rho_


# # Just for debugging
# def compute_predictor_stability_bound_0(gram_matrix, lam, dloss, test_prediction):
#     def bound(output_value):
#         loss_rho = 0.5 * np.abs(
#             dloss(output_value, test_prediction)
#             - dloss(np.zeros(test_prediction.shape), test_prediction)
#         )
#         return np.sqrt(gram_matrix[-1, -1]) * (loss_rho / (lam * gram_matrix.shape[0]))
#
#     return bound
# #


def compute_predictor_stability_bound_(gram_matrix, lam, loss_rho_, bigger_loss_rho_):
    def predictor_stability_bound_(output_value):
        predictor_stability_bound = np.sqrt(gram_matrix[-1, -1]) * np.minimum(
            bigger_loss_rho_(output_value) / ((lam**3) * (gram_matrix.shape[0] ** 2)),
            2 * loss_rho_(output_value) / (lam * gram_matrix.shape[0]),
        )
        return predictor_stability_bound

    return predictor_stability_bound_


def compute_scores_stability_bounds_(
    gram_matrix, non_conformity_rho, predictor_stability_bound_
):
    def bounds(output_value):
        return (
            np.sqrt(np.diag(gram_matrix))
            * non_conformity_rho
            * predictor_stability_bound_(output_value)
        )

    return bounds


def compute_approximate_test_prediction_(
    gram_matrix,
    dloss,
    proto_influence,
    test_prediction,
):
    def approximate_test_prediction_(output_value):
        proto_influence_test_prediction = (
            gram_matrix[:, -1].reshape(1, -1) @ proto_influence
        )
        influence_test_prediction_z = (
            (-1.0)
            * proto_influence_test_prediction
            * (
                dloss(np.zeros(test_prediction.shape), test_prediction)
                / gram_matrix.shape[0]
            )
        )
        influence_test_prediction_y = (
            (-1.0)
            * proto_influence_test_prediction
            * (dloss(output_value, test_prediction) / gram_matrix.shape[0])
        )
        return (
            test_prediction - influence_test_prediction_z + influence_test_prediction_y
        )

    return approximate_test_prediction_


def compute_upper_p_value_(
    train_scores_,
    scores_stability_bounds_,
    non_conformity,
    approximate_test_prediction_,
):
    def upper_p_value_(output_value):
        approximate_test_prediction = approximate_test_prediction_(output_value)
        scores_stability_bounds = scores_stability_bounds_(output_value)
        upper_train_scores = train_scores_(output_value) + scores_stability_bounds[:-1]
        p_value = (
            1
            + np.sum(
                upper_train_scores
                >= (
                    non_conformity(output_value, approximate_test_prediction)
                    - scores_stability_bounds[-1]
                )
            )
        ) / (upper_train_scores.shape[0] + 1)
        return p_value

    return upper_p_value_


def compute_lower_p_value_(
    train_scores_,
    scores_stability_bounds_,
    non_conformity,
    approximate_test_prediction_,
):

    def lower_p_value_(output_value):
        approximate_test_prediction = approximate_test_prediction_(output_value)
        scores_stability_bounds = scores_stability_bounds_(output_value)
        lower_train_scores = train_scores_(output_value) - scores_stability_bounds[:-1]
        p_value = (
            1
            + np.sum(
                lower_train_scores
                >= (
                    non_conformity(output_value, approximate_test_prediction)
                    + scores_stability_bounds[-1]
                )
            )
        ) / (lower_train_scores.shape[0] + 1)
        return p_value

    return lower_p_value_


def compute_big_loss_rho_bound(kernel_max, lam, sample_size_p1, loss_beta, loss_rho):
    return (1 + (kernel_max * loss_beta) / (lam * sample_size_p1)) * loss_rho


def compute_bigger_loss_rho_bound(
    kernel_max, lam, loss_beta, big_loss_rho_bound, loss_xi
):
    return (
        0.5 * loss_xi * (kernel_max**2) * (big_loss_rho_bound**2)
        + 2 * lam * kernel_max * loss_beta * big_loss_rho_bound
    )


def compute_scores_stability_bound(
    kernel_max, lam, sample_size_p1, loss_rho, bigger_loss_rho_bound
):
    return kernel_max * np.minimum(
        bigger_loss_rho_bound / ((lam**3) * (sample_size_p1**2)),
        2 * loss_rho / (lam * sample_size_p1),
    )


def compute_crude_thickness_upper_bound(
    kernel_max, lam, sample_size_p1, loss_rho, scores_stability_bound
):
    return 8 * (
        scores_stability_bound + (kernel_max * loss_rho / (lam * sample_size_p1))
    )


def compute_thicknes_upper_bound(
    kernel_max, lam, sample_size_p1, loss_beta, scores_stability_bound
):
    return (
        12 / (1 - (loss_beta * kernel_max / (lam * sample_size_p1)))
    ) * scores_stability_bound


class InfluenceFunctionConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "influence_function_cp"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]
        self.non_conformity_lams = non_conformity_["lams"]

        loss_ = maker(self.predictor.loss_name)(**self.predictor.loss_params)
        self.dloss = loss_["df"]
        self.loss_lams = loss_["lams"]

        empirical_risk = KernelEmpiricalRisk(
            self.predictor.loss_name, self.predictor.loss_params
        )
        self.gradient_hessian = empirical_risk.gradient_hessian

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

                self.predictor.fit(augmented_input_points, augmented_output_points)
                predictions = self.predictor.predict(augmented_input_points)

                gram_matrix = self.predictor._get_kernel(augmented_input_points)
                _, hessian = self.gradient_hessian(
                    self.predictor.model_weights,
                    gram_matrix,
                    augmented_output_points,
                    self.predictor.lam,
                )
                proto_influence, _, _, _ = np.linalg.lstsq(
                    hessian, gram_matrix[:, -1].reshape(-1, 1)
                )
                approximate_predictions_ = compute_approximate_predictions_(
                    gram_matrix, self.dloss, proto_influence, predictions
                )

                train_scores_ = compute_train_scores_(
                    self.non_conformity, train_output_points, approximate_predictions_
                )

                loss_rho_ = compute_loss_rho_(self.dloss, predictions[-1, :])
                big_loss_rho_ = compute_big_loss_rho_(
                    gram_matrix, self.predictor.lam, self.loss_lams["beta"], loss_rho_
                )
                bigger_loss_rho_ = compute_bigger_loss_rho_(
                    gram_matrix,
                    self.predictor.lam,
                    self.loss_lams["beta"],
                    big_loss_rho_,
                    self.loss_lams["xi"],
                )

                predictor_stability_bound_ = compute_predictor_stability_bound_(
                    gram_matrix, self.predictor.lam, loss_rho_, bigger_loss_rho_
                )

                scores_stability_bounds_ = compute_scores_stability_bounds_(
                    gram_matrix,
                    self.non_conformity_lams["rho"],
                    predictor_stability_bound_,
                )

                output_min = np.min(train_output_points)
                output_max = np.max(train_output_points)

                # # Just for debugging
                # ys = np.linspace(output_min, output_max, 100)
                # fig, ax = plt.subplots()
                # ax.plot(ys, loss_rho_(ys), label = "loss_rho")
                # ax.plot(ys, big_loss_rho_(ys), label = "big_loss_rho")
                # ax.plot(ys, bigger_loss_rho_(ys), label = "bigger_loss_rho")
                # ax.legend()
                # fig.show()

                # predictor_stability_bound_0 = compute_predictor_stability_bound_0(
                #     gram_matrix, self.predictor.lam, self.dloss, predictions[-1, :]
                # )
                # ys = np.linspace(output_min, output_max, 100)
                # fig, ax = plt.subplots()
                # ax.plot(ys, predictor_stability_bound_(ys), label = "infunc")
                # ax.plot(ys, predictor_stability_bound_0(ys), label = "stable")
                # ax.legend()
                # fig.show()
                # #

                approximate_test_prediction_ = compute_approximate_test_prediction_(
                    gram_matrix, self.dloss, proto_influence, predictions[-1, :]
                )

                upper_p_value_ = compute_upper_p_value_(
                    train_scores_,
                    scores_stability_bounds_,
                    self.non_conformity,
                    approximate_test_prediction_,
                )
                upper_prediction_region = inter_finder(
                    lambda output_value: (
                        upper_p_value_(output_value) - confidence_control_level
                    ),
                    output_min,
                    output_max,
                    predictions[-1, :].item(),
                )

                lower_p_value_ = compute_lower_p_value_(
                    train_scores_,
                    scores_stability_bounds_,
                    self.non_conformity,
                    approximate_test_prediction_,
                )
                lower_prediction_region = inter_finder(
                    lambda output_value: (
                        lower_p_value_(output_value) - confidence_control_level
                    ),
                    output_min,
                    output_max,
                    predictions[-1, :].item(),
                )

                prediction_regions.append(
                    {"upper": upper_prediction_region, "lower": lower_prediction_region}
                )

            return prediction_regions

        return region_predictor

    def thickness_upper_bound(self, train_input_points, test_input_points):
        """
        Computes theoretical upper bound on the thickness
        """
        upper_bounds = []
        for test_input_point in test_input_points:
            augmented_input_points = np.concatenate(
                (train_input_points, test_input_point.reshape(1, -1))
            )
            gram_matrix = self.predictor._get_kernel(augmented_input_points)

            kernel_max = np.max(np.diag(gram_matrix))
            sample_size_p1 = gram_matrix.shape[0]

            big_loss_rho_bound = compute_big_loss_rho_bound(
                kernel_max,
                self.predictor.lam,
                sample_size_p1,
                self.loss_lams["beta"],
                self.loss_lams["rho"],
            )
            bigger_loss_rho_bound = compute_bigger_loss_rho_bound(
                kernel_max,
                self.predictor.lam,
                self.loss_lams["beta"],
                big_loss_rho_bound,
                self.loss_lams["xi"],
            )
            scores_stability_bound = compute_scores_stability_bound(
                kernel_max,
                self.predictor.lam,
                sample_size_p1,
                self.loss_lams["rho"],
                bigger_loss_rho_bound,
            )

            if (self.predictor.lam * sample_size_p1) <= (
                kernel_max * self.loss_lams["beta"] * 0.5
            ):
                upper_bound = compute_crude_thickness_upper_bound(
                    kernel_max,
                    self.predictor.lam,
                    sample_size_p1,
                    self.loss_lams["rho"],
                    scores_stability_bound,
                )
            else:
                upper_bound = compute_thicknes_upper_bound(
                    kernel_max,
                    self.predictor.lam,
                    sample_size_p1,
                    self.loss_lams["beta"],
                    scores_stability_bound,
                )

            upper_bounds.append(upper_bound)

        return upper_bounds
