"""
The purpose of the present module is to build prediction sets using split conformal prediction provided a predictor.

It should be able to be used as follows:

```
train_input_points, test_input_points, train_output_points, test_output_points = (
    train_test_split(input_points, output_points, random_state=0)
)
(
    proper_train_input_points,
    calib_input_points,
    proper_train_output_points,
    calib_output_points,
) = train_test_split(train_input_points, train_output_points, random_state=0)

predictor = Regression()
predictor.fit(train_input_points, train_output_points)  # the model is responsible for its fitting

conformal_predictor = SplitConformalPredictor(predictor, non_conformity_name="absolute")
conformal_predictor.fit(calib_input_points, calib_output_points)
region_predictor = conformal_predictor.predict(test_input_points)

confidence_control_level = 0.1
prediction_regions = region_predictor(confidence_control_level)
```

"""

import numpy as np
import portion as P
from ..models.losses import maker


class SplitConformalPredictor:
    def __init__(
        self, predictor, non_conformity_name="absolute", non_conformity_params={}
    ):
        self.name = "scp"
        self.predictor = predictor
        non_conformity_ = maker(non_conformity_name)(**non_conformity_params)
        self.non_conformity = non_conformity_["f"]
        self.calibration_scores = None

    def fit(self, input_points, output_points):
        """
        Compute calibration scores
        """
        predictions = self.predictor.predict(input_points)
        self.calibration_scores = self.non_conformity(output_points, predictions)
        return

    def predict(self, input_points):
        """
        Prediction region function (as a function of the confidence level)
        for each input point
        """
        if self.calibration_scores is None:
            raise ValueError("Please compute the calibration points first.")

        def region_predictor(confidence_control_level):
            number_of_calibration_points = self.calibration_scores.size
            quantile_level = (
                np.ceil(
                    (number_of_calibration_points + 1) * (1 - confidence_control_level)
                )
                / number_of_calibration_points
            )
            quantile_value = np.quantile(
                self.calibration_scores, quantile_level, method="higher"
            )

            predictions = self.predictor.predict(input_points)
            prediction_regions = [
                P.closed(prediction - quantile_value, prediction + quantile_value)
                for prediction in predictions
            ]
            return prediction_regions

        return region_predictor
