import os
import shutil
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from surprise.surprise_adequacy import DSA
from surprise.surprise_adequacy import LSA
from surprise.surprise_adequacy import SurpriseAdequacyConfig


class TestSurpriseAdequacyConsistency(unittest.TestCase):

    def setUp(self) -> None:
        path = '/tmp/data/'
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        self.config = SurpriseAdequacyConfig(saved_path=path, is_classification=True, layer_names=['activation_3'],
                                             ds_name='mnist', num_classes=10, min_var_threshold=1e-5, batch_size=128)
        self.model: tf.keras.Model = load_model('./tests/assets/model_mnist.h5')
        (self.train_data, _), (self.test_data, y_test) = mnist.load_data()
        self.train_data = self.train_data.reshape(-1, 28, 28, 1)
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)

        self.train_data = self.train_data.astype("float32")
        self.train_data = (self.train_data / 255.0) - (1.0 - 0.5)
        self.test_data = self.test_data.astype("float32")
        self.test_data = (self.test_data / 255.0) - (1.0 - 0.5)

    def test_train_ats_calculation_against_kims_implementation(self):
        datasplit_train, datasplit_test = self.train_data, self.test_data

        # HERE you'll calculate the ats on your code
        nodes = 10
        sa = DSA(self.model, datasplit_train, config=self.config)
        ats, pred = sa._calculate_ats(datasplit_train)

        # Here you load the values from kims implementation
        kim_ats = np.load('./tests/assets/original_mnist_train_activation_3_ats.npy')

        kim_pred = np.load('./tests/assets/original_mnist_train_pred.npy')

        self.assertIsInstance(ats, np.ndarray)
        self.assertEqual(ats.shape, (60000, nodes))
        self.assertEqual(ats.dtype, np.float32)
        np.testing.assert_almost_equal(ats, kim_ats, decimal=5)

        self.assertIsInstance(pred, np.ndarray)
        self.assertEqual(pred.shape, (60000,))
        self.assertEqual(pred.dtype, np.int)
        np.testing.assert_equal(pred, kim_pred)

    def test_dsa_is_consistent_with_original_implementation(self):
        our_dsa = DSA(model=self.model, train_data=self.train_data, config=self.config)
        our_dsa.prep()
        test_dsa, predictions = our_dsa.calc(self.test_data, "test", use_cache=False)

        original_dsa = np.load("./tests/assets/original_dsa_scores.npy")

        np.testing.assert_almost_equal(actual=test_dsa,
                                       desired=original_dsa, decimal=2)

    def test_lsa_is_consistent_with_original_implementation(self):
        our_lsa = LSA(model=self.model, train_data=self.train_data, config=self.config)
        # train_ats, train_pred, kde, removed_rows will be overridden in next steps
        our_lsa.prep()

        our_lsa.train_ats = np.load("./tests/assets/original_mnist_train_activation_3_ats.npy")
        our_lsa.train_pred = np.load("./tests/assets/original_mnist_train_pred.npy")

        our_lsa._load_or_create_likelyhood_estimator(use_cache=False)
        from_original_test_ats = np.load("./tests/assets/mnist_test_activation_3_ats.npy")
        from_original_test_pred = np.load("./tests/assets/mnist_test_pred.npy")
        # Method under test
        our_lsa = our_lsa._calc_lsa(from_original_test_ats, from_original_test_pred)
        original_lsa = np.load("./tests/assets/original_lsa_scores.npy")

        np.testing.assert_almost_equal(actual=our_lsa,
                                       desired=original_lsa, decimal=2)

    def test_lsa_kdes(self):
        nodes = 10
        our_lsa = LSA(model=self.model, train_data=self.train_data, config=self.config)
        our_lsa.prep()
        test_kdes, test_rm_rows = our_lsa._calc_kdes()

        self.assertIsInstance(test_kdes, dict)
        self.assertIsInstance(test_rm_rows, list)
        self.assertEqual(len(test_kdes), nodes)
        if len(test_rm_rows) == 0:
            self.assertEqual(np.array(test_rm_rows).dtype, float)
        else:
            self.assertEqual(np.array(test_rm_rows).dtype, int)

    def test_output_dim_reduction(self):

        def original_implementation(layer_output):
            # This is the original dimensionality reduction implemented by Kim et al
            # (only thing we replaced is len(dataset) with layer_output.shape[0] and the array conversion)
            mapper = map(lambda x: [np.mean(x[..., j]) for j in range(x.shape[-1])],
                         [layer_output[i] for i in range(layer_output.shape[0])])
            return np.array([x for x in mapper])

        layer_outputs_1 = np.zeros(shape=(100, 25, 25, 3))
        expected = original_implementation(layer_outputs_1)
        actual = LSA._output_dim_reduction(layer_outputs_1)
        np.testing.assert_almost_equal(expected, actual)

        np.random.seed(0)
        shape = (100, 20, 20, 3)
        layer_outputs_2 = np.random.rand(np.prod(shape)).reshape(shape)
        expected = original_implementation(layer_outputs_2)
        actual = LSA._output_dim_reduction(layer_outputs_2)
        np.testing.assert_almost_equal(expected, actual)

        shape = (100, 10, 11, 12, 13)
        layer_outputs_3 = np.random.rand(np.prod(shape)).reshape(shape)
        expected = original_implementation(layer_outputs_3)
        actual = LSA._output_dim_reduction(layer_outputs_3)
        np.testing.assert_almost_equal(expected, actual)
