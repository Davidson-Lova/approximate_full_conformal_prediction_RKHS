"""
The purpose of the present module is to build prediction sets using oracle conformal prediction provided a predictor.

It should be able to be used as follows:
```

train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)

predictor = Regression()

conformal_predictor = OracleConformalPredictor(predictor, non_conformity_name="absolute")
region_predictor = conformal_predictor.fit_predict(
    train_input_points, train_output_points,
    test_input_points, test_output_points
)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
```
"""

import numpy as np
import portion as P
from ..models.losses import maker


class OracleConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "oracle"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]

    def fit_predict(
        self,
        train_input_points,
        train_output_points,
        test_input_points,
        test_output_points,
    ):
        """
        Prediction region function (as a function of the confidence level)
        for each test input point
        """

        number_of_train_points = train_input_points.shape[0]

        def region_predictor(confidence_control_level):
            prediction_regions = []
            for test_input_point, test_output_point in zip(
                test_input_points, test_output_points
            ):

                augmented_input_points = np.concatenate(
                    (train_input_points, test_input_point.reshape(1, -1))
                )
                augmented_output_points = np.concatenate(
                    (train_output_points, test_output_point.reshape(1, -1))
                )

                self.predictor.fit(augmented_input_points, augmented_output_points)
                predictions = self.predictor.predict(augmented_input_points)
                scores = self.non_conformity(augmented_output_points, predictions)

                quantile_value = np.quantile(scores, 1 - confidence_control_level, method="higher")

                prediction_regions.append(
                    P.closed(
                        predictions[-1, :] - quantile_value,
                        predictions[-1, :] + quantile_value,
                    )
                )

            return prediction_regions

        return region_predictor
