import pytest
from numpy import array, append
from numpy.testing import assert_allclose, assert_array_equal

from touvlo.rec_sys.cf import cost_function, grad, unravel_params


class TestCollaborativeeFiltering:

    @pytest.fixture(scope="module")
    def R(self):
        return array([[1, 1, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 0, 0, 0]])

    @pytest.fixture(scope="module")
    def Y(self):
        return array([[5, 4, 0, 0],
                      [3, 0, 0, 0],
                      [4, 0, 0, 0],
                      [3, 0, 0, 0],
                      [3, 0, 0, 0]])

    @pytest.fixture(scope="module")
    def X(self):
        return array([[1.048686, -0.400232, 1.194119],
                      [0.780851, -0.385626, 0.521198],
                      [0.641509, -0.547854, -0.083796],
                      [0.453618, -0.800218, 0.680481],
                      [0.937538, 0.106090, 0.361953]])

    @pytest.fixture(scope="module")
    def theta(self):
        return array([[0.28544, -1.68427, 0.26294],
                      [0.50501, -0.45465, 0.31746],
                      [-0.43192, -0.47880, 0.84671],
                      [0.72860, -0.27189, 0.32684]])

    def test_cost_function1(self, X, Y, R, theta):
        _lambda = 0
        assert_allclose(cost_function(X, Y, R, theta, _lambda),
                        22.225,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cost_function2(self, X, Y, R, theta):
        _lambda = 1.5
        assert_allclose(cost_function(X, Y, R, theta, _lambda),
                        31.344,
                        rtol=0, atol=0.001, equal_nan=False)

    def test_unravel_params(self, X, theta):
        num_users = 4
        num_products = 5
        num_features = 3
        flat = append(X.flatten(), theta.flatten())
        X_infltd, theta_infltd = unravel_params(flat, num_users,
                                                num_products, num_features)

        assert_array_equal(X_infltd, X)
        assert_array_equal(theta_infltd, theta)

    def test_grad1(self, X, Y, R, theta):
        num_users = 4
        num_products = 5
        num_features = 3
        _lambda = 0

        flat = append(X.flatten(), theta.flatten())
        params = grad(flat, Y, R, num_users, num_products,
                      num_features, _lambda)
        X_grad, theta_grad = unravel_params(params, num_users,
                                            num_products, num_features)

        assert_allclose(X_grad,
                        array([[-2.52899, 7.57570, -1.89979],
                               [-0.56820, 3.35265, -0.52340],
                               [-0.83241, 4.91163, -0.76678],
                               [-0.38358, 2.26334, -0.35334],
                               [-0.80378, 4.74272, -0.74041]]),
                        rtol=0, atol=0.001, equal_nan=False)
        assert_allclose(theta_grad,
                        array([[-10.56802, 4.62776, -7.16004],
                               [-3.05099, 1.16441, -3.47411],
                               [0.00000, 0.00000, 0.00000],
                               [0.00000, 0.00000, 0.00000]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_grad2(self, X, Y, R, theta):
        num_users = 4
        num_products = 5
        num_features = 3
        _lambda = 1.5

        flat = append(X.flatten(), theta.flatten())
        params = grad(flat, Y, R, num_users, num_products,
                      num_features, _lambda)
        X_grad, theta_grad = unravel_params(params, num_users,
                                            num_products, num_features)

        assert_allclose(X_grad,
                        array([[-0.95596, 6.97536, -0.10861],
                               [0.60308, 2.77421, 0.25840],
                               [0.12986, 4.08985, -0.89247],
                               [0.29684, 1.06301, 0.66738],
                               [0.60253, 4.90185, -0.19748]]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(theta_grad,
                        array([[-10.13985, 2.10136, -6.76564],
                               [-2.29347, 0.48244, -2.99791],
                               [-0.64787, -0.71821, 1.27007],
                               [1.09290, -0.40784, 0.49027]]),
                        rtol=0, atol=0.001, equal_nan=False)
