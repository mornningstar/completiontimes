from unittest import TestCase

import numpy as np

from src.predictions.baselines.median_base_model import MedianBaseModel


class TestMedianBaseModel(TestCase):

    def test_evaluate_returns_constant_array_matching_y_test(self):
        model = MedianBaseModel()
        y_train = np.array([10, 20, 30])
        model.train(np.array([[0]] * 3), y_train)

        x_test = np.array([[1], [2]])
        y_test = np.array([100, 200])
        eval_preds = model.evaluate(x_test, y_test)
        assert isinstance(eval_preds, np.ndarray)
        assert eval_preds.shape == y_test.shape
        assert np.all(eval_preds == np.median(y_train))

    def test_predict_returns_constant_array(self):
        model = MedianBaseModel()
        x_train = np.array([[1], [2], [3]])
        y_train = np.array([1, 4, 7])
        model.train(x_train, y_train)

        x_test = np.array([[5], [6], [7], [8]])
        preds = model.predict(x_test)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(x_test),)
        assert np.all(preds == np.median(y_train))
