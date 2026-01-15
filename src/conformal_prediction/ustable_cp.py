"""The purpose of the present module is to build prediction sets using conformal prediction for kernel ridge regression.

It should be able to be used as follows:
```

train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)

predictor = Regression()

conformal_predictor = UStableConformalPredictor(predictor, non_conformity_name="absolute")
region_predictor = conformal_predictor.fit_predict(train_input_points, train_output_points, test_input_points)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
"""

import numpy as np
import portion as P
from ..models.losses import maker


def commpute_predictor_stability_bound(gram_matrix, lam, loss_rho):
    return np.sqrt(gram_matrix[-1, -1]) * (loss_rho / (lam * gram_matrix.shape[0]))


def compute_scores_stability_bounds(
    gram_matrix, non_conformity_rho, predictor_stability_bound
):
    return (
        np.sqrt(np.diag(gram_matrix)) * non_conformity_rho * predictor_stability_bound
    )


class UStableConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "ustable_cp"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]
        self.non_conformity_lams = non_conformity_["lams"]

        loss_ = maker(self.predictor.loss_name)(**self.predictor.loss_params)
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
                predictor_stability_bound = commpute_predictor_stability_bound(
                    gram_matrix, self.predictor.lam, self.loss_lams["rho"]
                )
                scores_stability_bounds = compute_scores_stability_bounds(
                    gram_matrix,
                    self.non_conformity_lams["rho"],
                    predictor_stability_bound,
                )

                quantile_level = np.ceil(
                    (gram_matrix.shape[0]) * (1 - confidence_control_level)
                ) / (gram_matrix.shape[0] - 1)

                upper_train_scores = train_scores + scores_stability_bounds[:-1]
                upper_quantile_value = np.quantile(
                    upper_train_scores, quantile_level, method="higher"
                )
                upper_prediction_region = P.closed(
                    predictions[-1, :]
                    - upper_quantile_value
                    - scores_stability_bounds[-1],
                    predictions[-1, :]
                    + upper_quantile_value
                    + scores_stability_bounds[-1],
                )

                lower_train_scores = train_scores - scores_stability_bounds[:-1]
                lower_quantile_value = np.quantile(
                    lower_train_scores, quantile_level, method="higher"
                )
                lower_prediction_region = P.closed(
                    predictions[-1, :]
                    - lower_quantile_value
                    + scores_stability_bounds[-1],
                    predictions[-1, :]
                    + lower_quantile_value
                    - scores_stability_bounds[-1],
                )

                prediction_regions.append(
                    {"upper": upper_prediction_region, "lower": lower_prediction_region}
                )
            return prediction_regions

        return region_predictor
