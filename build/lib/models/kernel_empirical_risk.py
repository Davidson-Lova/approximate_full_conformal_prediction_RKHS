import numpy as np
from .losses import maker

def rkhs_norm(model_weights, gram_matrix):
    return np.sum(np.dot(model_weights.T, gram_matrix) * model_weights.T)


class KernelEmpiricalRisk:
    def __init__(self,
        loss_name="log_cosh",
        loss_params={}
    ):
        self.loss_name = loss_name
        self.loss_params = loss_params
        loss_ = maker(loss_name)
        self.loss = loss_["f"]
        self.dloss = loss_["df"]
        self.d2loss = loss_["ddf"]

    def empirical_risk(self, model_weights, gram_matrix, output_points, lam):
        """Computes the empirical risk

        Parameters
        ----------
        model_weights : ndarray of shape (number_of_points, number_of_outputs)
            or (number_of_points * number_of_outputs, )
            Model parameters

        gram_matrix : ndarray of shape (number_of_points, number_of_points)

        output_points : ndarray of shape (number_of_points, )

        lam : float
            The regularization parameter

        Returns
        -------
        empirical_risk_ : float
            Weighted average of losses per sample, plus penalty.
        """
        model_weights_is_flat = model_weights.ndim == 1
        if model_weights_is_flat:
            model_weights_size = model_weights.size
            number_of_points = gram_matrix.shape[0]
            number_of_outputs = int(model_weights_size / number_of_points)
            model_weights = model_weights.reshape(
                (number_of_points, number_of_outputs), order="F"
            )

        predictions = np.dot(gram_matrix, model_weights)
        empirical_risk = np.mean(
            [
                self.loss(output_point, prediction)
                for output_point, prediction in zip(output_points, predictions)
            ]
        ) + lam * rkhs_norm(model_weights, gram_matrix)

        return empirical_risk

    # def empirical_risk_gradient(self, model_weights, gram_matrix, output_points, lam):
    #     """Computes the empirical risk, and its gradient w.r.t. model_weights.

    #     Parameters
    #     ----------
    #     model_weights : ndarray of shape (number_of_points, number_of_outputs)
    #         or (number_of_points * number_of_outputs, )
    #         Model parameters

    #     gram_matrix : ndarray of shape (number_of_points, number_of_points)

    #     output_points : ndarray of shape (number_of_points, )

    #     lam : float
    #         The regularization parameter

    #     Returns
    #     -------
    #     empirical_risk : float
    #         Weighted average of losses per sample, plus penalty.

    #     gradient : ndarray of shape model_weights.shape
    #          The gradient of the loss.
    #     """
    #     model_weights_is_flat = model_weights.ndim == 1
    #     if model_weights_is_flat:
    #         model_weights_size = model_weights.size
    #         number_of_points = gram_matrix.shape[0]
    #         number_of_outputs = int(model_weights_size / number_of_points)
    #         model_weights = model_weights.reshape(
    #             (number_of_points, number_of_outputs), order="F"
    #         )

    #     predictions = np.dot(gram_matrix, model_weights)

    #     empirical_risk = np.mean(
    #         [
    #             self.loss(output_point, prediction)
    #             for output_point, prediction in zip(output_points, predictions)
    #         ]
    #     ) + lam * rkhs_norm(model_weights, gram_matrix)

    #     probability_matrix = softmax(predictions, axis=1)

    #     number_of_points, number_of_outputs = model_weights.shape

    #     one_hot_output_points = (
    #         output_points.reshape(-1, 1) == np.arange(number_of_outputs)
    #     ).astype(float)

    #     gradient = np.dot(
    #         gram_matrix,
    #         (probability_matrix - one_hot_output_points) / number_of_points
    #         + 2 * lam * model_weights,
    #     )
    #     gradient = gradient.ravel(order="F")

    #     return empirical_risk, gradient

    # def gradient(self, model_weights, gram_matrix, output_points, lam):
    #     """Computes the gradient of the empirical risk w.r.t. model_weights.

    #     Parameters
    #     ----------
    #     model_weights : ndarray of shape (number_of_points, number_of_outputs)
    #         or (number_of_points * number_of_outputs, )
    #         Model parameters

    #     gram_matrix : ndarray of shape (number_of_points, number_of_points)

    #     output_points : ndarray of shape (number_of_points, )

    #     lam : float
    #         The regularization parameter

    #     Returns
    #     -------
    #     gradient : ndarray of shape model_weights.shape
    #          The gradient of the loss.
    #     """
    #     model_weights_is_flat = model_weights.ndim == 1
    #     if model_weights_is_flat:
    #         model_weights_size = model_weights.size
    #         number_of_points = gram_matrix.shape[0]
    #         number_of_outputs = int(model_weights_size / number_of_points)
    #         model_weights = model_weights.reshape(
    #             (number_of_points, number_of_outputs), order="F"
    #         )

    #     predictions = np.dot(gram_matrix, model_weights)

    #     probability_matrix = softmax(predictions, axis=1)

    #     number_of_points, number_of_outputs = model_weights.shape

    #     one_hot_output_points = (
    #         output_points.reshape(-1, 1) == np.arange(number_of_outputs)
    #     ).astype(float)

    #     gradient = np.dot(
    #         gram_matrix,
    #         (probability_matrix - one_hot_output_points) / number_of_points
    #         + 2 * lam * model_weights,
    #     )
    #     gradient = gradient.ravel(order="F")

    #     return gradient

    # def hess(self, model_weights, gram_matrix, output_points, lam):
    #     """Computes the hessian of the empirical risk w.r.t. model_weights.

    #     Parameters
    #     ----------
    #     model_weights : ndarray of shape (number_of_points, number_of_outputs)
    #         Model parameters

    #     gram_matrix : ndarray of shape (number_of_points, number_of_points)

    #     output_points : ndarray of shape (number_of_points, )

    #     lam : float
    #         The regularization parameter

    #     Returns
    #     -------
    #     hess : ndarray of shape model_weights.shape
    #          The gradient of the loss.
    #     """
    #     return

    # def gradient_hessian_product(self, model_weights, gram_matrix, output_points, lam):
    #     """Computes the sum of loss and gradient w.r.t. model_weights.

    #     Parameters
    #     ----------
    #     model_weights : ndarray of shape (number_of_points, number_of_outputs)
    #         Model parameters

    #     gram_matrix : ndarray of shape (number_of_points, number_of_points)

    #     output_points : ndarray of shape (number_of_points, )

    #     lam : float
    #         The regularization parameter

    #     Returns
    #     -------
    #     loss : float
    #         Weighted average of losses per sample, plus penalty.

    #     gradient : ndarray of shape model_weights.shape
    #          The gradient of the loss.
    #     """
    #     return
