"""The purpose of the present module is to build prediction sets using local stable conformal prediction for kernel ridge regression.

It should be able to be used as follows:
```

train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)

predictor = Regression()

conformal_predictor = StableConformalPredictor(predictor, non_conformity_name="absolute")
region_predictor = conformal_predictor.fit_predict(train_input_points, train_output_points, test_input_points)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
"""

import numpy as np
from ..models.losses import maker
from .utils import inter_finder


def compute_predictor_stability_bound_(gram_matrix, lam, dloss, test_prediction):
    def bound(output_value):
        loss_rho = 0.5 * np.abs(
            dloss(output_value, test_prediction)
            - dloss(np.zeros(test_prediction.shape), test_prediction)
        )
        return np.sqrt(gram_matrix[-1, -1]) * (loss_rho / (lam * gram_matrix.shape[0]))

    return bound


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


def compute_upper_p_value_(
    train_scores, scores_stability_bounds_, non_conformity, test_prediction
):
    def upper_p_value_(output_value):
        scores_stability_bounds = scores_stability_bounds_(output_value)
        upper_train_scores = train_scores + scores_stability_bounds[:-1]
        p_value = (
            1
            + np.sum(
                upper_train_scores
                >= (
                    non_conformity(output_value, test_prediction)
                    - scores_stability_bounds[-1]
                )
            )
        ) / (upper_train_scores.shape[0] + 1)
        return p_value

    return upper_p_value_


def compute_lower_p_value_(
    train_scores, scores_stability_bounds_, non_conformity, test_prediction
):

    def lower_p_value_(output_value):
        scores_stability_bounds = scores_stability_bounds_(output_value)
        lower_train_scores = train_scores - scores_stability_bounds[:-1]
        p_value = (
            1
            + np.sum(
                lower_train_scores
                >= (
                    non_conformity(output_value, test_prediction)
                    + scores_stability_bounds[-1]
                )
            )
        ) / (lower_train_scores.shape[0] + 1)
        return p_value

    return lower_p_value_


def compute_thickness_upper_bound(gram_matrix, lam, loss_rho):
    return (8 * loss_rho * np.max(np.diag(gram_matrix))) / (lam * gram_matrix.shape[0])


class StableConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "stable_cp"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]
        self.non_conformity_lams = non_conformity_["lams"]

        loss_ = maker(self.predictor.loss_name)(**self.predictor.loss_params)
        self.dloss = loss_["df"]
        self.loss_lams = loss_["lams"]

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
                train_scores = self.non_conformity(
                    train_output_points,
                    predictions[:-1, :].reshape(train_output_points.shape),
                ).flatten()

                gram_matrix = self.predictor._get_kernel(augmented_input_points)
                predictor_stability_bound_ = compute_predictor_stability_bound_(
                    gram_matrix, self.predictor.lam, self.dloss, predictions[-1, :]
                )
                scores_stability_bounds_ = compute_scores_stability_bounds_(
                    gram_matrix,
                    self.non_conformity_lams["rho"],
                    predictor_stability_bound_,
                )

                output_min = np.min(train_output_points)
                output_max = np.max(train_output_points)

                upper_p_value_ = compute_upper_p_value_(
                    train_scores,
                    scores_stability_bounds_,
                    self.non_conformity,
                    predictions[-1, :],
                )
                upper_prediction_region = inter_finder(
                    lambda output_value: (
                        upper_p_value_(output_value) - confidence_control_level
                    ),
                    output_min,
                    output_max,
                    predictions[-1, :],
                )

                lower_p_value_ = compute_lower_p_value_(
                    train_scores,
                    scores_stability_bounds_,
                    self.non_conformity,
                    predictions[-1, :],
                )
                lower_prediction_region = inter_finder(
                    lambda output_value: (
                        lower_p_value_(output_value) - confidence_control_level
                    ),
                    output_min,
                    output_max,
                    predictions[-1, :],
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
            upper_bound = compute_thickness_upper_bound(
                gram_matrix, self.predictor.lam, self.loss_lams["rho"]
            )
            upper_bounds.append(upper_bound)

        return upper_bounds
