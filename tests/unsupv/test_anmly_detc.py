import pytest
from numpy import array
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal)

from touvlo.unsupv.anmly_detc import (estimate_uni_gaussian, uni_gaussian,
                                      estimate_multi_gaussian, cov_matrix,
                                      multi_gaussian, is_anomaly, predict)


class TestAnomalyDetection:

    @pytest.fixture
    def X(self):
        return array([[13.047, 14.741], [13.409, 13.763], [14.196, 15.853],
                      [14.915, 16.174], [13.577, 14.043], [13.922, 13.406],
                      [12.822, 14.223], [15.676, 15.892], [16.163, 16.203],
                      [12.666, 14.899], [13.985, 12.958], [14.061, 14.549],
                      [13.390, 15.562], [13.394, 15.627], [13.979, 13.281],
                      [14.168, 14.466], [13.962, 14.752], [14.459, 15.070],
                      [14.585, 15.827], [12.074, 13.067], [13.549, 15.538],
                      [13.986, 14.788], [14.970, 16.518], [14.256, 15.294],
                      [15.334, 16.125]])

    def test_is_anomaly1(self):
        prob = array([[0.7], [0.8], [0.2], [0.9], [0.3]])
        threshold = 0.35

        assert_array_equal(is_anomaly(prob, threshold),
                           array([[0], [0], [1], [0], [1]]))

    def test_is_anomaly2(self):
        prob = array([[0.47], [0.28], [0.32], [0.79], [0.13]])
        threshold = 0.4

        assert_array_equal(is_anomaly(prob, threshold),
                           array([[0], [1], [1], [0], [1]]))

    def test_predict1(self, X):
        sigma = array([[0.83945644, 0.56925192],
                       [0.56925192, 1.06358832]])
        mu = array([14.022, 14.905]),
        epsilon = 0.1

        assert_array_equal(predict(X, epsilon, multi_gaussian,
                                   mu=mu, sigma=sigma),
                           array([[1], [0], [0], [1], [0], [1], [1], [1],
                                  [1], [1], [1], [0], [1], [1], [1], [0],
                                  [0], [0], [0], [1], [1], [0], [1], [0],
                                  [1]]))

    def test_predict2(self, X):
        sigma2 = array([0.83944, 1.06363])
        mu = array([14.022, 14.905]),
        epsilon = 0.1

        assert_array_equal(predict(X, epsilon, uni_gaussian,
                                   mu=mu, sigma2=sigma2),
                           array([[1], [1], [0], [1], [0], [1], [1], [1],
                                  [1], [1], [1], [0], [0], [0], [1], [0],
                                  [0], [0], [1], [1], [0], [0], [1], [0],
                                  [1]]))

    def test_estimate_uni_gaussian(self, X):
        mu, sigma2 = estimate_uni_gaussian(X)

        assert_allclose(mu,
                        array([14.022, 14.905]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(sigma2,
                        array([0.83944, 1.06363]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_estimate_multi_gaussian(self, X):
        mu, sigma = estimate_multi_gaussian(X)

        assert_allclose(mu,
                        array([14.022, 14.905]),
                        rtol=0, atol=0.001, equal_nan=False)

        assert_allclose(sigma,
                        array([[0.83945644, 0.56925192],
                               [0.56925192, 1.06358832]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_uni_gaussian1(self):
        mu = array([14.0218])
        sigma2 = array([0.8394564])
        X = array([[13.047]])
        assert_allclose(uni_gaussian(X, mu, sigma2),
                        array([[0.24723]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_uni_gaussian2(self):
        mu = array([14, 13, 17])
        sigma2 = array([0.4, 0.3, 0.2])
        X = array([[13.8, 12.8, 16.7], [13.6, 13, 17.4], [13.4, 13.4, 16.4]])
        assert_allclose(uni_gaussian(X, mu, sigma2),
                        array([[0.29123], [0.22493], [0.081380]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_uni_gaussian3(self, X):
        mu = array([14.022, 14.905])
        sigma2 = array([0.83944, 1.06363])

        assert_allclose(uni_gaussian(X, mu, sigma2),
                        array([[0.0944222], [0.0729627], [0.1083855],
                               [0.0491118], [0.1055603], [0.0582816],
                               [0.0574507], [0.0208647], [0.0049715],
                               [0.0563965], [0.0283335], [0.1585582],
                               [0.1083846], [0.1041962], [0.0486835],
                               [0.1519022], [0.1662344], [0.1483876],
                               [0.0934671], [0.0035961], [0.1221021],
                               [0.1672274], [0.0289997], [0.1518109],
                               [0.0299930]]),
                        rtol=0, atol=0.001, equal_nan=False)

    def test_cov_matrix1(self, X):
        m, n = X.shape
        mu = array([14.022, 14.905])
        sigma = cov_matrix(X, mu)

        assert sigma.shape == (n, n)
        assert sigma[0, 1] == sigma[1, 0]
        assert_almost_equal(sigma[0, 0], 0.83944, decimal=4)
        assert_almost_equal(sigma[1, 1], 1.06363, decimal=4)

    def test_cov_matrix2(self, X):
        R = array([[1, 2, 3], [2, 4, 1], [3, 1, 1], [4, 1, 2]])
        m, n = R.shape

        mu = array([2.5, 2, 1.75])
        sigma = cov_matrix(R, mu)

        assert sigma.shape == (n, n)
        assert sigma[0, 1] == sigma[1, 0]
        assert sigma[0, 2] == sigma[2, 0]
        assert sigma[1, 2] == sigma[2, 1]

        assert_almost_equal(sigma[0, 0], 1.25, decimal=4)
        assert_almost_equal(sigma[1, 1], 1.5, decimal=4)
        assert_almost_equal(sigma[2, 2], 0.6875, decimal=4)

    def test_multi_gaussian(self, X):
        mu = array([14.022, 14.905])
        sigma = array([[0.83945644, 0.56925192], [0.56925192, 1.06358832]])

        assert_allclose(multi_gaussian(X, mu, sigma),
                        array([[0.09982199], [0.11430868], [0.12466088],
                               [0.09484268], [0.14877596], [0.04627195],
                               [0.08836677], [0.04082006], [0.0135213],
                               [0.03812894], [0.01381082], [0.18926753],
                               [0.06971848], [0.06310562], [0.03226401],
                               [0.16829367], [0.20862722], [0.18595913],
                               [0.140872], [0.01807496], [0.09439509],
                               [0.20953774], [0.06169397], [0.19641747],
                               [0.06984078]]),
                        rtol=0, atol=0.001, equal_nan=False)
