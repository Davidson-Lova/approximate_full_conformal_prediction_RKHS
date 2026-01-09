"Conformal prediction"


class cp:
    def __init__(self, predictor, non_conformity_maker, non_conformity_params):
        self.predictor = predictor
        self.non_conformity_maker = non_conformity_maker
        self.non_conformity_params = non_conformity_params

        self.non_conformity_bundle = self.non_conformity_maker(
            **self.non_conformity_params
        )

    def update_non_conformity(self, non_conformity_params):
        self.non_conformity_bundle = self.non_conformity_maker(**non_conformity_params)

    def set_predictor_param(self, _a_):
        self.predictor._a_ = _a_
        return

    def fit(self, X_eval, X_rep, Y_eval, params):
        self.predictor.fit(X_eval, X_rep, Y_eval, **params)
        return

    def predict(self, X_eval, X_rep):
        res = self.predictor.predict(X_eval, X_rep)
        return res

    def compute_ncs(self, X_eval, X_rep, y):
        y_pred = self.predict(X_eval, X_rep).reshape(y.shape)
        non_conformity_scores = self.non_conformity_bundle["f"](y, y_pred)
        return non_conformity_scores
