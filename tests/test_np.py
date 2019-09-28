import unittest

from numpy import array, append
from numpy.testing import assert_allclose

from ml_algorithms.nn import (g_grad, cost_function, grad,
                              unravel_params, back_propagation,
                              feed_forward)


class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        pass

    def test_sigmoid_grad(self):
        z = array([-1, -0.5, 0, 0.5, 1])
        assert_allclose(g_grad(z),
                        [0.196612, 0.235004, 0.25, 0.235004, 0.196612],
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function(self):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        theta1 = array([[0.35, 0.78, 0.13, 0.90], [0.27, 0.66, 0.62, 0.20],
                        [0.64, 0.36, 0.76, 0.33], [0.00, 0.70, 0.78, 0.85],
                        [0.55, 0.72, 0.24, 0.43]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])
        _lambda = 0
        num_labels = 3

        assert_allclose(cost_function(X, y, theta1, theta2,
                                      _lambda, num_labels),
                        4.2549,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward1(self):
        X = array([[1, 0.10, 0.30, -0.50]])
        theta1 = array([[0.35, 0.78, 0.13, 0.90], [0.27, 0.66, 0.62, 0.20],
                        [0.64, 0.36, 0.76, 0.33], [0.00, 0.70, 0.78, 0.85],
                        [0.55, 0.72, 0.24, 0.43]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])

        a1, a2, a3, z2 = feed_forward(X, theta1, theta2)

        assert_allclose(a1,
                        array([[1, 0.10, 0.30, -0.50]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a2,
                        array([[1, 0.50425, 0.60396, 0.67678,
                                0.46979, 0.61751]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a3,
                        array([[0.91672, 0.85806, 0.81484]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z2,
                        array([[0.017, 0.422, 0.739, -0.121, 0.479]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward2(self):
        X = array([[1, -0.2, 0, -0.6]])
        theta1 = array([[0.35, 0.78, 0.13, 0.90], [0.27, 0.66, 0.62, 0.20],
                        [0.64, 0.36, 0.76, 0.33], [0.00, 0.70, 0.78, 0.85],
                        [0.55, 0.72, 0.24, 0.43]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])

        a1, a2, a3, z2 = feed_forward(X, theta1, theta2)

        assert_allclose(a1,
                        array([[1, -0.2, 0, -0.6]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a2,
                        array([[1, 0.41435, 0.5045, 0.59146,
                                0.34299, 0.53693]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a3,
                        array([[0.89115, 0.83518, 0.78010]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z2,
                        array([[-0.346, 0.018, 0.37, -0.65, 0.148]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward3(self):
        X = array([[1, 0, 0.2, 0.45]])
        theta1 = array([[0.35, 0.78, 0.13, 0.90], [0.27, 0.66, 0.62, 0.20],
                        [0.64, 0.36, 0.76, 0.33], [0.00, 0.70, 0.78, 0.85],
                        [0.55, 0.72, 0.24, 0.43]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])

        a1, a2, a3, z2 = feed_forward(X, theta1, theta2)

        assert_allclose(a1,
                        array([[1, 0, 0.2, 0.45]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a2,
                        array([[1, 0.6859, 0.61869, 0.7192,
                                0.63146, 0.68815]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a3,
                        array([[0.9388, 0.88464, 0.84335]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z2,
                        array([[0.781, 0.484, 0.9405, 0.5385, 0.7915]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad(self):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        theta1 = array([[0.35, 0.78, 0.13, 0.90], [0.27, 0.66, 0.62, 0.20],
                        [0.64, 0.36, 0.76, 0.33], [0.00, 0.70, 0.78, 0.85],
                        [0.55, 0.72, 0.24, 0.43]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])
        _lambda = 0
        num_labels = 3
        input_layer_size = 3
        hidden_layer_size = 5
        nn_params = append(theta1.flatten(), theta2.flatten())
        nn_grad = grad(nn_params, X, y, _lambda, input_layer_size,
                       num_labels, hidden_layer_size)
        grad1, grad2 = unravel_params(nn_grad, input_layer_size,
                                      hidden_layer_size, num_labels)

        assert_allclose(grad1,
                        array([[0.2331544, -0.0131348,
                                0.0334961, -0.0652458],
                               [0.1224948, -0.0088256,
                                0.0156733, -0.0124328],
                               [0.1457463, -0.0012316,
                                0.0279176, -0.0223957],
                               [0.2254230, -0.0137763,
                                0.0313083, -0.0402217],
                               [0.1379756, 0.0072703,
                                0.0348654, -0.0063072]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(grad2,
                        array([
                            [0.58222, 0.32373, 0.32671,
                             0.38197, 0.28645, 0.35770],
                            [0.52596, 0.23320, 0.28940,
                             0.33057, 0.20557, 0.29964],
                            [0.47943, 0.29941, 0.30099,
                             0.34265, 0.27997, 0.32182]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_1(self):
        h = array([[0.91672, 0.85806, 0.81484], [0.89115, 0.83518, 0.78010],
                   [0.93880, 0.88464, 0.84335]])
        z2 = array([[0.017, 0.422, 0.739, -0.121, 0.479],
                    [-0.346, 0.018, 0.37, -0.65, 0.148],
                    [0.781, 0.484, 0.9405, 0.5385, 0.7915]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])
        y = array([[0]])
        num_labels = 3

        delta2, delta3 = back_propagation(theta2, h, z2, y, num_labels)

        assert_allclose(delta2,
                        array([[0.205846, 0.043161, 0.165794,
                                0.141024, 0.216743],
                               [0.188319, 0.038974, 0.173862,
                                0.117159, 0.218117],
                               [0.187209, 0.047684, 0.159976,
                                0.141731, 0.204305]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta3,
                        array([[-0.083275, 0.858059, 0.814840],
                               [-0.108852, 0.835179, 0.780102],
                               [-0.061198, 0.884641, 0.843350]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_2(self):
        h = array([[0.91672, 0.85806, 0.81484], [0.89115, 0.83518, 0.78010],
                   [0.93880, 0.88464, 0.84335]])
        z2 = array([[0.017, 0.422, 0.739, -0.121, 0.479],
                    [-0.346, 0.018, 0.37, -0.65, 0.148],
                    [0.781, 0.484, 0.9405, 0.5385, 0.7915]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])
        y = array([[2]])
        num_labels = 3

        delta2, delta3 = back_propagation(theta2, h, z2, y, num_labels)

        assert_allclose(delta2,
                        array([[0.32083735, 0.15318951, 0.10016938,
                                0.31787549, 0.00889491],
                               [0.29994511, 0.15396509, 0.10137133,
                                0.27715581, -0.00068307],
                               [0.28631281, 0.15620337, 0.09939030,
                                0.30696021, 0.01545845]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta3,
                        array([[0.91672, 0.85806, -0.18516],
                               [0.89115, 0.83518, -0.21990],
                               [0.93880, 0.88464, -0.15665]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_3(self):
        h = array([[0.91672, 0.85806, 0.81484], [0.89115, 0.83518, 0.78010],
                   [0.93880, 0.88464, 0.84335]])
        z2 = array([[0.017, 0.422, 0.739, -0.121, 0.479],
                    [-0.346, 0.018, 0.37, -0.65, 0.148],
                    [0.781, 0.484, 0.9405, 0.5385, 0.7915]])
        theta2 = array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                        [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                        [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])
        y = array([[1]])
        num_labels = 3

        delta2, delta3 = back_propagation(theta2, h, z2, y, num_labels)

        assert_allclose(delta2,
                        array([[0.21335, 0.16754, 0.17673, 0.26557, 0.20966],
                               [0.19560, 0.16896, 0.18594, 0.22983, 0.21066],
                               [0.19367, 0.17036, 0.17007, 0.25809, 0.19787]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta3,
                        array([[0.91672, -0.14194, 0.81484],
                               [0.89115, -0.16482, 0.78010],
                               [0.93880, -0.11536, 0.84335]]),
                        rtol=0, atol=0.001, equal_nan=False)
