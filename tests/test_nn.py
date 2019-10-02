import pytest
from numpy import array, append, empty, zeros, int64
from numpy.testing import assert_allclose

from ml_algorithms.nn import (feed_forward, init_nn_weights,
                              back_propagation, cost_function,
                              grad, unravel_params)


class TestNeuralNetwork:

    @pytest.fixture(scope="module")
    def omicron(self):
        return array([[0.35, 0.78, 0.13, 0.90],
                      [0.27, 0.66, 0.62, 0.20],
                      [0.64, 0.36, 0.76, 0.33],
                      [0.00, 0.70, 0.78, 0.85],
                      [0.55, 0.72, 0.24, 0.43]])

    @pytest.fixture(scope="module")
    def omega(self):
        return array([[0.86, 0.77, 0.63, 0.35, 0.99, 0.11],
                      [0.84, 0.74, 0.11, 0.30, 0.49, 0.14],
                      [0.04, 0.31, 0.17, 0.65, 0.28, 0.99]])

    @pytest.fixture(scope="module")
    def kappa(self):
        return array([[0.98, 0.6, 0.18, 0.47, 0.07, 1],
                      [0.9, 0.38, 0.38, 0.36, 0.52, 0.85],
                      [0.57, 0.23, 0.41, 0.45, 0.04, 0.24],
                      [0.46, 0.94, 0.03, 0.06, 0.19, 0.63],
                      [0.87, 0.4, 0.85, 0.07, 0.81, 0.76]])

    @pytest.fixture(scope="module")
    def upsilon(self):
        return array([[0.9, 0.95, 0.05, 0.05, 0.65, 0.11],
                      [0.84, 0.57, 0.17, 0.62, 0.06, 0.36],
                      [0.36, 0.6, 0.54, 0.49, 0.3, 0.03],
                      [0.11, 0.49, 0.71, 0.43, 0.01, 0.92],
                      [0.02, 0.01, 0.94, 0.35, 0.69, 0.88]])

    @pytest.fixture(scope="module")
    def zeta(self):
        return array([[0.91672, 0.85806, 0.81484],
                      [0.89115, 0.83518, 0.78010],
                      [0.93880, 0.88464, 0.84335]])

    @pytest.fixture(scope="module")
    def iota(self):
        return array([[0.017, 0.422, 0.739, -0.121, 0.479],
                      [-0.346, 0.018, 0.37, -0.65, 0.148],
                      [0.781, 0.484, 0.9405, 0.5385, 0.7915]])

    def test_cost_function1(self, omicron, omega):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 1
        num_labels = 3
        _lambda = 0

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        4.2549,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function2(self, omicron, omega):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 1
        num_labels = 3
        _lambda = 1

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        5.9738,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function3(self, omicron, omega):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 1
        num_labels = 3
        _lambda = 10

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        21.443,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function4(self, omicron, omega, kappa, upsilon):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 3
        num_labels = 3
        _lambda = 0

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = kappa
        theta[2] = upsilon
        theta[3] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        5.6617,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function5(self, omicron, omega, kappa, upsilon):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 3
        num_labels = 3
        _lambda = 1

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = kappa
        theta[2] = upsilon
        theta[3] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        9.7443,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function6(self, omicron, omega, kappa, upsilon):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])
        n_hidden_layers = 3
        num_labels = 3
        _lambda = 10

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = kappa
        theta[2] = upsilon
        theta[3] = omega

        assert_allclose(cost_function(X, y, theta, _lambda,
                                      num_labels, n_hidden_layers),
                        46.488,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward1(self, omicron, omega):
        n_hidden_layers = 1
        X = array([[1, 0.10, 0.30, -0.50]])

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        z, a = feed_forward(X, theta)

        assert_allclose(a[0],
                        array([[1, 0.10, 0.30, -0.50]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[1],
                        array([[1, 0.50425, 0.60396, 0.67678,
                                0.46979, 0.61751]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[2],
                        array([[0.91672, 0.85806, 0.81484]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z[1],
                        array([[0.017, 0.422, 0.739, -0.121, 0.479]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward2(self, omicron, omega):
        n_hidden_layers = 1
        X = array([[1, -0.2, 0, -0.6]])

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        z, a = feed_forward(X, theta)

        assert_allclose(a[0],
                        array([[1, -0.2, 0, -0.6]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[1],
                        array([[1, 0.41435, 0.5045, 0.59146,
                                0.34299, 0.53693]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[2],
                        array([[0.89115, 0.83518, 0.78010]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z[1],
                        array([[-0.346, 0.018, 0.37, -0.65, 0.148]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward3(self, omicron, omega):
        n_hidden_layers = 1
        X = array([[1, 0, 0.2, 0.45]])
        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = omicron
        theta[1] = omega

        z, a = feed_forward(X, theta, n_hidden_layers)
        assert_allclose(a[0],
                        array([[1, 0, 0.2, 0.45]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[1],
                        array([[1, 0.6859, 0.61869, 0.7192,
                                0.63146, 0.68815]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(a[2],
                        array([[0.9388, 0.88464, 0.84335]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z[1],
                        array([[0.781, 0.484, 0.9405, 0.5385, 0.7915]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(z[2],
                        array([[2.73048142, 2.03713759, 1.68336723]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_feed_forward4(self, omicron, omega, kappa, upsilon):
        n_hidden_layers = 3
        X = array([[1, 0, 0.2, 0.45]])
        theta = empty((n_hidden_layers + 1), dtype=object)

        theta[0] = omicron
        theta[1] = kappa
        theta[2] = upsilon
        theta[3] = omega

        z, a = feed_forward(X, theta, n_hidden_layers)

        assert_allclose(a[0],
                        array([[1, 0, 0.2, 0.45]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(a[1],
                        array([[1, 0.6859, 0.61869,
                                0.7192, 0.63146, 0.68815]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(a[2],
                        array([[1, 0.92912, 0.92877,
                                0.8169, 0.84812, 0.9402]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(a[3],
                        array([[1, 0.92585, 0.91859,
                                0.89109, 0.92052, 0.93092]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(a[4],
                        array([[0.97003, 0.92236, 0.90394]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(z[1],
                        array([[0.781, 0.484, 0.9405, 0.5385, 0.7915]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(z[2],
                        array([[2.5733, 2.5679, 1.4955, 1.72, 2.7551]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(z[3],
                        array([[2.5247, 2.4233, 2.1019, 2.4494, 2.6008]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(z[4],
                        array([[3.4772, 2.4749, 2.2417]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad(self, omicron, omega):
        X = array([[0.10, 0.30, -0.50], [-0.20, 0, -0.60], [0, 0.20, 0.45]])
        y = array([[0], [2], [1]])

        _lambda = 0
        num_labels = 3
        n_hidden_layers = 1
        input_layer_size = 3
        hidden_layer_size = 5

        theta = empty((n_hidden_layers + 1), dtype=object)

        theta[0] = omicron
        theta[1] = omega

        nn_params = append(theta[0].flatten(), theta[1].flatten())
        for i in range(2, len(theta)):
            nn_params = append(nn_params, theta[i].flatten())

        theta_grad = grad(X, y, nn_params, _lambda, input_layer_size,
                          hidden_layer_size, num_labels, n_hidden_layers)

        theta_grad = unravel_params(theta_grad, input_layer_size,
                                    hidden_layer_size, num_labels,
                                    n_hidden_layers)

        assert_allclose(theta_grad[0],
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

        assert_allclose(theta_grad[1],
                        array([
                            [0.58222, 0.32373, 0.32671,
                             0.38197, 0.28645, 0.35770],
                            [0.52596, 0.23320, 0.28940,
                             0.33057, 0.20557, 0.29964],
                            [0.47943, 0.29941, 0.30099,
                             0.34265, 0.27997, 0.32182]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_1(self, omega, zeta, iota):
        num_labels = 3
        y = array([[0]])
        n_hidden_layers = 1
        L = n_hidden_layers + 1  # last layer

        theta = empty((n_hidden_layers + 1), dtype=object)
        a = empty((n_hidden_layers + 2), dtype=object)
        z = empty((n_hidden_layers + 2), dtype=object)

        theta[1] = omega
        a[2] = zeta
        z[1] = iota

        delta = back_propagation(y, theta, a, z, num_labels, n_hidden_layers)

        assert_allclose(delta[L - 1],
                        array([[0.205846, 0.043161, 0.165794,
                                0.141024, 0.216743],
                               [0.188319, 0.038974, 0.173862,
                                0.117159, 0.218117],
                               [0.187209, 0.047684, 0.159976,
                                0.141731, 0.204305]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta[L],
                        array([[-0.083275, 0.858059, 0.814840],
                               [-0.108852, 0.835179, 0.780102],
                               [-0.061198, 0.884641, 0.843350]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_2(self, omega, zeta, iota):
        num_labels = 3
        y = array([[2]])
        n_hidden_layers = 1
        L = n_hidden_layers + 1  # last layer

        theta = empty((n_hidden_layers + 1), dtype=object)
        a = empty((n_hidden_layers + 2), dtype=object)
        z = empty((n_hidden_layers + 2), dtype=object)

        theta[1] = omega
        a[2] = zeta
        z[1] = iota

        delta = back_propagation(y, theta, a, z, num_labels, n_hidden_layers)

        assert_allclose(delta[L - 1],
                        array([[0.32083735, 0.15318951, 0.10016938,
                                0.31787549, 0.00889491],
                               [0.29994511, 0.15396509, 0.10137133,
                                0.27715581, -0.00068307],
                               [0.28631281, 0.15620337, 0.09939030,
                                0.30696021, 0.01545845]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta[L],
                        array([[0.91672, 0.85806, -0.18516],
                               [0.89115, 0.83518, -0.21990],
                               [0.93880, 0.88464, -0.15665]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_3(self, omega, zeta, iota):
        num_labels = 3
        y = array([[1]])
        n_hidden_layers = 1
        L = n_hidden_layers + 1  # last layer

        theta = empty((n_hidden_layers + 1), dtype=object)
        a = empty((n_hidden_layers + 2), dtype=object)
        z = empty((n_hidden_layers + 2), dtype=object)

        theta[1] = omega
        a[2] = zeta
        z[1] = iota

        delta = back_propagation(y, theta, a, z, num_labels, n_hidden_layers)

        assert_allclose(delta[L - 1],
                        array([[0.21335, 0.16754, 0.17673, 0.26557, 0.20966],
                               [0.19560, 0.16896, 0.18594, 0.22983, 0.21066],
                               [0.19367, 0.17036, 0.17007, 0.25809, 0.19787]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(delta[L],
                        array([[0.91672, -0.14194, 0.81484],
                               [0.89115, -0.16482, 0.78010],
                               [0.93880, -0.11536, 0.84335]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_back_prop_4(self, omicron, omega, kappa, upsilon):
        num_labels = 3
        y = array([[1]])
        n_hidden_layers = 3
        L = n_hidden_layers + 1  # last layer

        theta = empty((n_hidden_layers + 1), dtype=object)
        z = empty((n_hidden_layers + 2), dtype=object)
        a = empty((n_hidden_layers + 2), dtype=object)

        theta[0] = omicron
        theta[1] = kappa
        theta[2] = upsilon
        theta[3] = omega

        a[0] = array([[1, 0, 0.2, 0.45]])
        a[1] = array([[1, 0.6859, 0.61869, 0.7192, 0.63146, 0.68815]])
        a[2] = array([[1, 0.92912, 0.92877, 0.8169, 0.84812, 0.9402]])
        a[3] = array([[1, 0.92585, 0.91859, 0.89109, 0.92052, 0.93092]])
        a[4] = array([[0.97003, 0.92236, 0.90394]])

        z[1] = array([[0.781, 0.484, 0.9405, 0.5385, 0.7915]])
        z[2] = array([[2.5733, 2.5679, 1.4955, 1.72, 2.7551]])
        z[3] = array([[2.5247, 2.4233, 2.1019, 2.4494, 2.6008]])
        z[4] = array([[3.4772, 2.4749, 2.2417]])

        delta = back_propagation(y, theta, a, z, num_labels, n_hidden_layers)

        assert_allclose(delta[L],
                        array([[0.970032, -0.077638, 0.903935]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(delta[L - 1],
                        array([[0.066569, 0.056555, 0.087710,
                                0.085996, 0.063716]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(delta[L - 2],
                        array([[0.0125701, 0.0119912, 0.0210376,
                                0.0151738, 0.0093052]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(delta[L - 3],
                        array([[0.0075239, 0.0056169, 0.0042922,
                                0.0042767, 0.0095374]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_init_nn_weights1(self):
        num_labels = 3
        n_hidden_layers = 2
        input_layer_size = 3
        hidden_layer_size = 5

        theta = init_nn_weights(input_layer_size, hidden_layer_size,
                                num_labels, n_hidden_layers)

        assert theta[0].shape == (5, 4)
        assert theta[1].shape == (5, 6)
        assert theta[2].shape == (3, 6)
        assert len(theta) == n_hidden_layers + 1

    def test_init_nn_weights2(self):
        num_labels = 10
        n_hidden_layers = 5
        input_layer_size = 50
        hidden_layer_size = 25

        theta = init_nn_weights(input_layer_size, hidden_layer_size,
                                num_labels, n_hidden_layers)

        assert theta[0].shape == (25, 51)
        assert theta[1].shape == (25, 26)
        assert theta[2].shape == (25, 26)
        assert theta[3].shape == (25, 26)
        assert theta[4].shape == (25, 26)
        assert theta[5].shape == (10, 26)
        assert len(theta) == n_hidden_layers + 1

    def test_unravel_params1(self):
        num_labels = 3
        n_hidden_layers = 1
        input_layer_size = 3
        hidden_layer_size = 5

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = zeros((hidden_layer_size, (input_layer_size + 1)),
                         dtype=int64)
        theta[1] = zeros((num_labels, (hidden_layer_size + 1)),
                         dtype=int64)

        flat = append(theta[0].flatten(), theta[1].flatten())

        inflated = unravel_params(flat, input_layer_size, hidden_layer_size,
                                  num_labels, n_hidden_layers)

        assert inflated[0].shape == theta[0].shape
        assert inflated[1].shape == theta[1].shape
        assert len(inflated) == n_hidden_layers + 1

    def test_unravel_params2(self):
        num_labels = 3
        n_hidden_layers = 2
        input_layer_size = 3
        hidden_layer_size = 5

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = zeros((hidden_layer_size, (input_layer_size + 1)),
                         dtype=int64)
        theta[1] = zeros((hidden_layer_size, (hidden_layer_size + 1)),
                         dtype=int64)
        theta[2] = zeros((num_labels, (hidden_layer_size + 1)),
                         dtype=int64)

        flat = append(theta[0].flatten(), theta[1].flatten())
        flat = append(flat, theta[2].flatten())

        inflated = unravel_params(flat, input_layer_size, hidden_layer_size,
                                  num_labels, n_hidden_layers)

        assert inflated[0].shape == theta[0].shape
        assert inflated[1].shape == theta[1].shape
        assert inflated[2].shape == theta[2].shape
        assert len(inflated) == n_hidden_layers + 1

    def test_unravel_params3(self):
        num_labels = 10
        n_hidden_layers = 5
        input_layer_size = 50
        hidden_layer_size = 25

        theta = empty((n_hidden_layers + 1), dtype=object)
        theta[0] = zeros((hidden_layer_size, (input_layer_size + 1)),
                         dtype=int64)
        theta[1] = zeros((hidden_layer_size, (hidden_layer_size + 1)),
                         dtype=int64)
        theta[2] = zeros((hidden_layer_size, (hidden_layer_size + 1)),
                         dtype=int64)
        theta[3] = zeros((hidden_layer_size, (hidden_layer_size + 1)),
                         dtype=int64)
        theta[4] = zeros((hidden_layer_size, (hidden_layer_size + 1)),
                         dtype=int64)
        theta[5] = zeros((num_labels, (hidden_layer_size + 1)),
                         dtype=int64)

        flat = append(theta[0].flatten(), theta[1].flatten())
        flat = append(flat, theta[2].flatten())
        flat = append(flat, theta[3].flatten())
        flat = append(flat, theta[4].flatten())
        flat = append(flat, theta[5].flatten())

        inflated = unravel_params(flat, input_layer_size, hidden_layer_size,
                                  num_labels, n_hidden_layers)

        assert inflated[0].shape == theta[0].shape
        assert inflated[1].shape == theta[1].shape
        assert inflated[2].shape == theta[2].shape
        assert inflated[3].shape == theta[3].shape
        assert inflated[4].shape == theta[4].shape
        assert inflated[5].shape == theta[5].shape
        assert len(inflated) == n_hidden_layers + 1
