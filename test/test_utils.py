"""
Making sure that
each function in utils.utils does as wanted
"""

import json
import sys
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.append("..")
from predictive_region.utils import utils


class TestUtils(unittest.TestCase):
    def test_solveh_im(self):
        with open("test/params/utils.json", "r") as file:
            params = json.load(file)

        A = np.random.normal(
            0, 1, (params["solveh_im"]["n_samples"], params["solveh_im"]["n_features"])
        )
        scaler = StandardScaler()
        scaler.fit(A)
        A = scaler.transform(A)

        norm_gram_matrix = (
            np.matmul(A, A.transpose()) / params["solveh_im"]["n_samples"]
        )

        x_true = np.random.normal(0, 1, (params["solveh_im"]["n_samples"], 1))
        x_im_A = np.matmul(norm_gram_matrix, x_true)
        b = np.matmul(norm_gram_matrix, x_im_A)

        # b is surely in the image of the normalized gram matrix
        # So is x_im_A
        # Let us try to get x_im_A back
        x_im_A_approx = utils.solveh_im(
            norm_gram_matrix, b, params["solveh_im"]["rtol"]
        )

        self.assertAlmostEqual(
            np.linalg.norm(x_im_A - x_im_A_approx), 0.0, 5, "approximation bad"
        )

    def test_smallest_non_zero_eig(self):
        with open("test/params/utils.json", "r") as file:
            params = json.load(file)

        A = np.random.normal(
            0, 1, (params["smallest"]["n_samples"], params["solveh_im"]["n_features"])
        )
