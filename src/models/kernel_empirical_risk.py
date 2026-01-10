import numpy as np
from .losses import maker


def rkhs_norm(model_weights, gram_matrix):
    return np.sum(np.dot(model_weights.T, gram_matrix) * model_weights.T)


def compute_emp_risk(loss, output_points, predictions, gram_matrix, lam, model_weights):
    return np.mean(
        [
            loss(output_point, prediction)
            for output_point, prediction in zip(output_points, predictions)
        ]
    ) + lam * rkhs_norm(model_weights, gram_matrix)


def compute_grad_emp_risk(dloss, output_points, predictions, gram_matrix, lam):
    return (
        np.mean(
            [
                gram_vec.reshape(-1, 1) @ dloss(output_point, prediction).reshape(1, -1)
                for gram_vec, output_point, prediction in zip(
                    gram_matrix, output_points, predictions
                )
            ],
            axis=0,
        )
        + 2 * lam * predictions
    )


class KernelEmpiricalRisk:
    def __init__(self, loss_name="log_cosh", loss_params={"gamma": 1.0}):
        self.loss_name = loss_name
        print(self.loss_name)
        loss_ = maker(loss_name)(**loss_params)
        self.loss = loss_["f"]
        self.dloss = loss_["df"]
        self.d2loss = loss_["ddf"]

    def empirical_risk(self, model_weights, gram_matrix, output_points, lam):
        """Computes the regularized empirical risk

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
            model_weights = model_weights.reshape(output_points.shape, order="F")

        predictions = gram_matrix @ model_weights
        emp_risk = compute_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam, model_weights
        )

        return emp_risk

    def empirical_risk_gradient(self, model_weights, gram_matrix, output_points, lam):
        """Computes the regularized empirical risk, and its gradient w.r.t. model_weights.

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
        empirical_risk : float
            Weighted average of losses per sample, plus penalty.

        gradient : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.
        """
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(output_points.shape, order="F")

        predictions = gram_matrix @ model_weights

        emp_risk = compute_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam, model_weights
        )

        grad = compute_grad_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam
        )
        grad = grad.ravel(order="F")

        return emp_risk, grad

    def gradient(self, model_weights, gram_matrix, output_points, lam):
        """Computes the gradient of the regularized empirical risk w.r.t. model_weights.

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
        gradient : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.
        """
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(output_points.shape, order="F")

        predictions = gram_matrix @ model_weights

        grad = compute_grad_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam
        )
        grad = grad.ravel(order="F")

        return grad

    def gradient_hessian(self, model_weights, gram_matrix, output_points, lam):
        """Computes the gradient and the hessian of the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : ndarray of shape (number_of_points, number_of_outputs)
            Model parameters

        gram_matrix : ndarray of shape (number_of_points, number_of_points)

        output_points : ndarray of shape (number_of_points, )

        lam : float
            The regularization parameter

        Returns
        -------
        hess : ndarray of shape (model_weights.shape, model_weights.shape)
             The gradient of the regularized empirical risk.
        """
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(output_points.shape, order="F")

        predictions = gram_matrix @ model_weights

        grad = grad = compute_grad_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam
        )
        grad = grad.ravel(order="F")

        hess = np.mean(
            [
                np.kron(
                    self.d2loss(output_point, prediction).reshape(
                        output_points.shape[1], output_points.shape[1]
                    ),
                    gram_vec.reshape(-1, 1) @ gram_vec.reshape(1, -1),
                )
                for gram_vec, output_point, prediction in zip(
                    gram_matrix, output_points, predictions
                )
            ],
            axis=0,
        ) + 2 * lam * np.kron(np.eye(output_points.shape[1]), gram_matrix)

        return grad, hess

    def gradient_hessian_product(self, model_weights, gram_matrix, output_points, lam):
        """Computes the gradient and the hessian vector product function of
        the regularized empirical risk w.r.t. model_weights.

        Parameters
        ----------
        model_weights : ndarray of shape (number_of_points, number_of_outputs)
            Model parameters

        gram_matrix : ndarray of shape (number_of_points, number_of_points)

        output_points : ndarray of shape (number_of_points, )

        lam : float
            The regularization parameter

        Returns
        -------

        grad : ndarray of shape model_weights.shape
             The gradient of the regularized empirical risk.

        hessp : callable
            The hessian vector production function of the regularized empirical risk.
        """
        if model_weights.ndim == 1:
            model_weights = model_weights.reshape(output_points.shape, order="F")

        predictions = gram_matrix @ model_weights

        grad = compute_grad_emp_risk(
            self.dloss, output_points, predictions, gram_matrix, lam
        )
        grad = grad.ravel(order="F")

        def hessp(vec):
            if vec.ndim == 1:
                vec = vec.reshape(output_points.shape, order="F")

            hess_vec_prod = np.mean(
                [
                    gram_vec.reshape(-1, 1)
                    @ (
                        (gram_vec.reshape(1, -1) @ vec)
                        @ self.d2loss(output_point, prediction).reshape(
                            output_points.shape[1], output_points.shape[1]
                        )
                    )
                    for gram_vec, output_point, prediction in zip(
                        gram_matrix, output_points, predictions
                    )
                ],
                axis=0,
            ) + 2 * lam * (gram_matrix @ vec)

            hess_vec_prod = hess_vec_prod.ravel(order="F")

            return hess_vec_prod

        return grad, hessp
