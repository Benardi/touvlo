from numpy import array
from numpy.testing import assert_allclose

from touvlo.nnet.cmpt_grf import (init_params, linear_forward,
                                  linear_activation_forward,
                                  L_model_forward, L_model_backward,
                                  linear_backward, compute_cost,
                                  linear_activation_backward,
                                  update_parameters)


class TestNeuralNetwork:

    def test_init_params_1(self):
        parameters = init_params([3, 2, 1])

        assert parameters['W1'].shape == (2, 3)
        assert parameters['b1'].shape == (2, 1)
        assert parameters['W2'].shape == (1, 2)
        assert parameters['b2'].shape == (1, 1)

    def test_init_params_2(self):
        parameters = init_params([5, 4, 3, 1])

        assert parameters['W1'].shape == (4, 5)
        assert parameters['b1'].shape == (4, 1)
        assert parameters['W2'].shape == (3, 4)
        assert parameters['b2'].shape == (3, 1)
        assert parameters['W3'].shape == (1, 3)
        assert parameters['b3'].shape == (1, 1)

    def test_init_params_3(self):
        parameters = init_params([10, 7, 5, 2, 1])

        assert parameters['W1'].shape == (7, 10)
        assert parameters['b1'].shape == (7, 1)
        assert parameters['W2'].shape == (5, 7)
        assert parameters['b2'].shape == (5, 1)
        assert parameters['W3'].shape == (2, 5)
        assert parameters['b3'].shape == (2, 1)
        assert parameters['W4'].shape == (1, 2)
        assert parameters['b4'].shape == (1, 1)

    def test_linear_forward1(self):
        A = array([[1.62434536, -0.61175641], [-0.52817175, -1.07296862],
                   [0.86540763, -2.3015387]])
        W = array([[1.74481176, -0.7612069, 0.3190391]])
        b = array([[-0.24937038]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[3.26295, -1.23430]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_forward2(self):
        A = array([[-0.00172428, -0.00877858],
                   [0.00042214, 0.00582815],
                   [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_forward3(self):
        A = array([[0.00838983, 0.00931102],
                   [0.00285587, 0.00885141],
                   [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        Z, linear_cache = linear_forward(A, W, b)

        assert_allclose(Z, array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(linear_cache[0], A)
        assert_allclose(linear_cache[1], W)
        assert_allclose(linear_cache[2], b)

    def test_linear_activation_forward1(self):
        A_prev = array([[-0.41675785, -0.05626683],
                        [-2.1361961, 1.64027081],
                        [-1.79343559, -0.84174737]])
        W = array([[0.50288142, -1.24528809, -1.05795222]])
        b = array([[-0.90900761]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.96890023, 0.11013289]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[3.43896131, -2.08938436]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward2(self):
        A_prev = array([[-0.41675785, -0.05626683],
                        [-2.1361961, 1.64027081],
                        [-1.79343559, -0.84174737]])
        W = array([[0.50288142, -1.24528809, -1.05795222]])
        b = array([[-0.90900761]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[3.43896131, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[3.43896131, -2.08938436]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward3(self):
        A_prev = array([[-0.00172428, -0.00877858],
                        [0.00042214, 0.00582815],
                        [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.49826254, 0.498304]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward4(self):
        A_prev = array([[-0.00172428, -0.00877858],
                        [0.00042214, 0.00582815],
                        [-0.01100619, 0.01144724]])
        W = array([[0.00901591, 0.00502494, 0.00900856]])
        b = array([[-0.00683728]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[0, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00695, -0.006784]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward5(self):
        A_prev = array([[0.00838983, 0.00931102],
                        [0.00285587, 0.00885141],
                        [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="sigmoid")

        assert_allclose(A, array([[0.4997951, 0.49980945]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_forward6(self):
        A_prev = array([[0.00838983, 0.00931102],
                        [0.00285587, 0.00885141],
                        [-0.00754398, 0.01252868]])
        W = array([[0.0051293, -0.00298093, 0.00488518]])
        b = array([[-0.00075572]])

        A, cache = linear_activation_forward(A_prev, W, b,
                                             activation="relu")

        assert_allclose(A, array([[0, 0]]),
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(cache[0][0], A_prev)
        assert_allclose(cache[0][1], W)
        assert_allclose(cache[0][2], b)
        assert_allclose(cache[1], array([[-0.00075, -0.00067]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_L_model_forward1(self):
        parameters = {}
        parameters['W1'] = array([[0.35480861, 1.81259031, -1.3564758,
                                   -0.46363197, 0.82465384],
                                  [-1.17643148, 1.56448966, 0.71270509,
                                   -0.1810066, 0.53419953],
                                  [-0.58661296, -1.48185327, 0.85724762,
                                   0.94309899, 0.11444143],
                                  [-0.02195668, -2.12714455, -0.83440747,
                                   -0.46550831, 0.23371059]])
        parameters['b1'] = array([[1.38503523],
                                  [-0.51962709],
                                  [-0.78015214],
                                  [0.95560959]])
        parameters['W2'] = array([[-0.12673638, -1.36861282, 1.21848065,
                                   -0.85750144],
                                  [-0.56147088, -1.0335199, 0.35877096,
                                   1.07368134],
                                  [-0.37550472, 0.39636757, -0.47144628,
                                   2.33660781]])
        parameters['b2'] = array([[1.50278553], [-0.59545972], [0.52834106]])
        parameters['W3'] = array([[0.9398248, 0.42628539, -0.75815703]])
        parameters['b3'] = array([[-0.16236698]])

        X = array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                   [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                   [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                   [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                   [0.07612761, -0.15512816, 0.63422534, 0.810655]])

        Z1 = array([[-5.23825714, 3.18040136, 0.4074501, -1.88612721],
                    [-2.77358234, -0.56177316, 3.18141623, -0.99209432],
                    [4.18500916, -1.78006909, -0.14502619, 2.72141638],
                    [5.05850802, -1.25674082, -3.54566654, 3.82321852]])

        A1 = array([[0., 3.18040136, 0.4074501, 0.],
                    [0., 0., 3.18141623, 0.],
                    [4.18500916, 0., 0., 2.72141638],
                    [5.05850802, 0., 0., 3.82321852]])

        Z2 = array([[2.2644603, 1.09971298, -2.90298027, 1.54036335],
                    [6.33722569, -2.38116246, -4.11228806, 4.48582383],
                    [10.37508342, -0.66591468, 1.63635185, 8.17870169]])

        A2 = array([[2.2644603, 1.09971298, 0., 1.54036335],
                    [6.33722569, 0., 0., 4.48582383],
                    [10.37508342, 0., 1.63635185, 8.17870169]])

        Z3 = array([[-3.19864676, 0.87117055, -1.40297864, -3.00319435]])

        AL, caches = L_model_forward(X, parameters)

        assert_allclose(AL,
                        array([[0.03921668, 0.70498921,
                                0.19734387, 0.04728177]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        # caches = (cache1, cache2, cache3)
        assert len(caches) == 3

        # cache1 = ((X, W1, b1), Z1)
        assert_allclose(caches[0][0][0], X)
        assert_allclose(caches[0][0][1], parameters['W1'])
        assert_allclose(caches[0][0][2], parameters['b1'])
        assert_allclose(caches[0][1], Z1)

        # cache2 = ((A1, W2, b2), Z2)
        assert_allclose(caches[1][0][0], A1)
        assert_allclose(caches[1][0][1], parameters['W2'])
        assert_allclose(caches[1][0][2], parameters['b2'])
        assert_allclose(caches[1][1], Z2)

        # cache3 = ((A2, W3, b3), Z3)
        assert_allclose(caches[2][0][0], A2)
        assert_allclose(caches[2][0][1], parameters['W3'])
        assert_allclose(caches[2][0][2], parameters['b3'])
        assert_allclose(caches[2][1], Z3)

    def test_L_model_forward2(self):
        parameters = {}
        parameters['W1'] = array([[0.0072681, 0.00444083, -0.00856823,
                                   0.00446928, -0.01014648],
                                  [-0.02132323, 0.00173863, 0.00951201,
                                   0.00441897, 0.01469017],
                                  [0.01749516, 0.00353531, -0.00643337,
                                   -0.00047237, -0.0144904],
                                  [-0.0003619, -0.00090847, 0.0017629,
                                   0.0109462, -0.02126475]])
        parameters['b1'] = array([[0.00751449], [-0.00540607],
                                  [0.00793222], [0.00173653]])
        parameters['W2'] = array([[-0.01035434, 0.00874268, -0.00739572,
                                   0.00522945],
                                  [-0.00591876, -0.00477487, 0.0011253,
                                   0.01904742],
                                  [0.00694153, -0.00019581, 0.01662843,
                                   0.00030608]])
        parameters['b2'] = array([[-0.00297499], [-0.00968138],
                                  [0.00167067]])
        parameters['W3'] = array([[0.00116602, -0.00682257, -0.01914021]])
        parameters['b3'] = array([[-0.00139902]])

        X = array([[-0.31178367, 0.72900392, 0.21782079, -0.8990918],
                   [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
                   [1.63929108, -0.4298936, 2.63128056, 0.60182225],
                   [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
                   [0.07612761, -0.15512816, 0.63422534, 0.810655]])

        Z1 = array([[-0.02211435, 0.02765779, -0.01438117, -0.01854866],
                    [0.01214561, -0.02026147, 0.02674556, 0.02933695],
                    [-0.01780464, 0.02834375, -0.01044313, -0.02882979],
                    [0.00170298, 0.01673248, -0.00699771, -0.01132628]])

        A1 = array([[0., 0.02765779, 0., 0.],
                    [0.01214561, 0., 0.02674556, 0.02933695],
                    [0., 0.02834375, 0., 0.],
                    [0.00170298, 0.01673248, 0., 0.]])

        Z2 = array([[-0.0028599, -0.00338349, -0.00274116, -0.00271851],
                    [-0.00970693, -0.00949447, -0.00980908, -0.00982146],
                    [0.00166881, 0.00233909, 0.00166543, 0.00166492]])

        A2 = array([[0., 0., 0., 0.],
                    [0., 0., 0., 0.],
                    [0.00166881, 0.00233909, 0.00166543, 0.00166492]])

        Z3 = array([[-0.00143096, -0.00144379, -0.0014309, -0.00143089]])

        AL, caches = L_model_forward(X, parameters)

        assert_allclose(AL,
                        array([[0.49964226, 0.49963905,
                                0.49964228, 0.49964228]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        # caches = (cache1, cache2, cache3)
        assert len(caches) == 3

        # cache1 = ((X, W1, b1), Z1)
        assert_allclose(caches[0][0][0], X)
        assert_allclose(caches[0][0][1], parameters['W1'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[0][0][2], parameters['b1'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[0][1], Z1,
                        rtol=0, atol=0.0001, equal_nan=False)

        # cache2 = ((A1, W2, b2), Z2)
        assert_allclose(caches[1][0][0], A1,
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[1][0][1], parameters['W2'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[1][0][2], parameters['b2'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[1][1], Z2,
                        rtol=0, atol=0.0001, equal_nan=False)

        # cache3 = ((A2, W3, b3), Z3)
        assert_allclose(caches[2][0][0], A2,
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[2][0][1], parameters['W3'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[2][0][2], parameters['b3'],
                        rtol=0, atol=0.0001, equal_nan=False)
        assert_allclose(caches[2][1], Z3,
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_compute_cost1(self):
        AL = array([[0.8, 0.9, 0.4]])
        Y = array([[1, 1, 0]])

        assert compute_cost(AL, Y) == 0.2797765635793422

    def test_compute_cost2(self):
        AL = array([[0.3, 0.9, 0.5, 0.7]])
        Y = array([[0, 1, 1, 0]])

        assert compute_cost(AL, Y) == 0.5897888611206099

    def test_compute_cost3(self):
        AL = array([[0.2, 0.9, 0.8, 0.3, 0.1]])
        Y = array([[0, 1, 1, 0, 0]])

        assert compute_cost(AL, Y) == 0.20273661557656092

    def test_linear_backward1(self):
        dZ = array([[1.62434536, -0.61175641, -0.52817175, -1.07296862],
                    [0.86540763, -2.3015387, 1.74481176, -0.7612069],
                    [0.3190391, -0.24937038, 1.46210794, -2.06014071]])
        A_prev = array([
            [-0.3224172, -0.38405435, 1.13376944, -1.09989127],
            [-0.17242821, -0.87785842, 0.04221375, 0.58281521],
            [-1.10061918, 1.14472371, 0.90159072, 0.50249434],
            [0.90085595, -0.68372786, -0.12289023, -0.93576943],
            [-0.26788808, 0.53035547, -0.69166075, -0.39675353]])
        W = array([
            [-0.6871727, -0.84520564, -0.67124613, -0.0126646, -1.11731035],
            [0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896],
            [-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548]])
        b = array([[2.10025514], [0.12015895], [0.61720311]])

        dA_prev, dW, db = linear_backward(dZ, (A_prev, W, b))

        assert_allclose(
            dA_prev,
            array([[-1.15171336, 0.06718465, -0.3204696, 2.09812712],
                   [0.60345879, -3.72508701, 5.81700741, -3.84326836],
                   [-0.4319552, -1.30987417, 1.72354705, 0.05070578],
                   [-0.38981415, 0.60811244, -1.25938424, 1.47191593],
                   [-2.52214926, 2.67882552, -0.67947465, 1.48119548]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            dW,
            array([[0.07313866, -0.0976715, -0.87585828,
                    0.73763362, 0.00785716],
                   [0.85508818, 0.37530413, -0.59912655,
                    0.71278189, -0.58931808],
                   [0.97913304, -0.24376494, -0.08839671,
                    0.55151192, -0.10290907]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            db,
            array([[-0.14713786], [-0.11313155], [-0.13209101]]),
            rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_backward2(self):
        dZ = array([[0.00186561, 0.00410052, 0.001983, 0.00119009],
                    [-0.00670662, 0.00377564, 0.00121821, 0.01129484],
                    [0.01198918, 0.00185156, -0.00375285, -0.0063873]])
        A_prev = array([[0.00423494, 0.0007734, -0.00343854, 0.00043597],
                        [-0.00620001, 0.00698032, -0.00447129, 0.01224508],
                        [0.00403492, 0.00593579, -0.01094912, 0.00169382],
                        [0.00740556, -0.00953701, -0.00266219, 0.00032615],
                        [-0.01373117, 0.00315159, 0.00846161, -0.00859516]])
        W = array([
            [0.00350546, -0.01312283, -0.00038696, -0.01615772, 0.01121418],
            [0.00408901, -0.00024617, -0.00775162, 0.01273756, 0.01967102],
            [-0.01857982, 0.01236164, 0.01627651, 0.00338012, -0.01199268]])
        b = array([[0.00863345], [-0.0018092], [-0.00603921]])

        dA_prev, dW, db = linear_backward(dZ, (A_prev, W, b))

        assert_allclose(
            dA_prev,
            array([[-2.43640350e-04, -4.58892745e-06, 8.16598584e-05,
                    1.69031409e-04],
                   [1.25374740e-04, -3.18514742e-05, -7.27138058e-05,
                    -9.73553082e-05],
                   [2.46407217e-04, -7.17013089e-07, -7.12937314e-05,
                    -1.91976770e-04],
                   [-7.50452536e-05, -1.19040969e-05, -2.92087342e-05,
                    1.03049760e-04],
                   [-2.54787160e-04, 9.80493391e-05, 9.12078897e-05,
                    3.12127713e-04]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            dW,
            array([[1.19308584e-06, 5.69056250e-06, 3.04277686e-06,
                    -7.54542258e-06, -1.53588645e-06],
                   [-6.18669066e-06, 5.01988695e-05, 2.85957687e-07,
                    -2.13084891e-05, 4.30404291e-06],
                   [1.55812859e-05, -3.07103668e-05, 2.24093111e-05,
                    1.97589620e-05, -3.39113378e-05]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            db,
            array([[0.0022848], [0.00239552], [0.00092515]]),
            rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_backward1(self):
        dAL = array([[-0.41675785, -0.05626683]])
        A = array([[-2.1361961, 1.64027081],
                   [-1.79343559, -0.84174737],
                   [0.50288142, -1.24528809]])
        W = array([[-1.05795222, -0.90900761, 0.55145404]])
        b = array([[2.29220801]])
        Z = array([[0.04153939, -1.11792545]])

        dA_prev, dW, db = linear_activation_backward(dAL,
                                                     ((A, W, b), Z),
                                                     activation="sigmoid")

        assert_allclose(dA_prev,
                        array([[0.11017994, 0.01105339],
                               [0.09466817, 0.00949723],
                               [-0.05743092, -0.00576154]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(dW,
                        array([[0.10266786, 0.09778551, -0.01968084]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(db,
                        array([[-0.05729622]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_backward2(self):
        dAL = array([[-0.41675785, -0.05626683]])
        A = array([[-2.1361961, 1.64027081],
                   [-1.79343559, -0.84174737],
                   [0.50288142, -1.24528809]])
        W = array([[-1.05795222, -0.90900761, 0.55145404]])
        b = array([[2.29220801]])
        Z = array([[0.04153939, -1.11792545]])

        dA_prev, dW, db = linear_activation_backward(dAL,
                                                     ((A, W, b), Z),
                                                     activation="relu")

        assert_allclose(dA_prev,
                        array([[0.44090989, 0.],
                               [0.37883606, 0.],
                               [-0.2298228, 0.]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(dW,
                        array([[0.44513824, 0.37371418, -0.10478989]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(db,
                        array([[-0.20837892]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_backward3(self):
        dAL = array([[-0.00637655, -0.01187612]])
        A = array([[-0.01421217, -0.00153495],
                   [-0.00269057, 0.02231367],
                   [-0.02434768, 0.00112727]])
        W = array([[0.00370445, 0.01359634, 0.00501857]])
        b = array([[-0.00844214]])
        Z = array([[9.76147160e-08, 5.42352572e-03]])

        dA_prev, dW, db = linear_activation_backward(dAL,
                                                     ((A, W, b), Z),
                                                     activation="sigmoid")

        assert_allclose(dA_prev,
                        array([[-5.90539539e-06, -1.09985312e-05],
                               [-2.16744337e-05, -4.03676502e-05],
                               [-8.00029409e-06, -1.49001850e-05]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(dW,
                        array([[1.36067216e-05,
                                -3.09801701e-05,
                                1.77333419e-05]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(db,
                        array([[-0.00228157]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_linear_activation_backward4(self):
        dAL = array([[-0.00637655, -0.01187612]])
        A = array([[-0.01421217, -0.00153495],
                   [-0.00269057, 0.02231367],
                   [-0.02434768, 0.00112727]])
        W = array([[0.00370445, 0.01359634, 0.00501857]])
        b = array([[-0.00844214]])
        Z = array([[9.76147160e-08, 5.42352572e-03]])

        dA_prev, dW, db = linear_activation_backward(dAL,
                                                     ((A, W, b), Z),
                                                     activation="relu")

        assert_allclose(dA_prev,
                        array([[-2.36215816e-05, -4.39944483e-05],
                               [-8.66977348e-05, -1.61471788e-04],
                               [-3.20011763e-05, -5.96011785e-05]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(dW,
                        array([[5.44269535e-05,
                                -1.23921655e-04,
                                7.09333184e-05]]),
                        rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(db,
                        array([[-0.00912634]]),
                        rtol=0, atol=0.0001, equal_nan=False)

    def test_L_model_backward1(self):
        AL = array([[1.78862847, 0.43650985]])
        Y_assess = array([[1, 0]])

        A0 = array([[0.09649747, -1.8634927],
                    [-0.2773882, -0.35475898],
                    [-0.08274148, -0.62700068],
                    [-0.04381817, -0.47721803]])
        W1 = array([[-1.31386475, 0.88462238, 0.88131804, 1.70957306],
                    [0.05003364, -0.40467741, -0.54535995, -1.54647732],
                    [0.98236743, -1.10106763, -1.18504653, -0.2056499]])
        b1 = array([[1.48614836], [0.23671627], [-1.02378514]])
        Z1 = array([[-0.7129932, 0.62524497],
                    [-0.16051336, -0.76883635],
                    [-0.23003072, 0.74505627]])

        A1 = array([[1.97611078, -1.24412333],
                    [-0.62641691, -0.80376609],
                    [-2.41908317, -0.92379202]])
        W2 = array([[-1.02387576, 1.12397796, -0.13191423]])
        b2 = array([[-1.62328545]])
        Z2 = array([[0.64667545, -0.35627076]])

        grads = L_model_backward(AL, Y_assess,
                                 [((A0, W1, b1), Z1), ((A1, W2, b2), Z2)])

        assert_allclose(
            grads['dA0'],
            array([[0., 0.52257901],
                   [0., -0.3269206],
                   [0., -0.32070404],
                   [0., -0.74079187]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dW1'],
            array([[0.41010002, 0.07807203, 0.13798444, 0.10502167],
                   [0., 0., 0., 0.],
                   [0.05283652, 0.01005865, 0.01777766, 0.0135308]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['db1'],
            array([[-0.22007063], [0.], [-0.02835349]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dA1'],
            array([[0.12913162, -0.44014127],
                   [-0.14175655, 0.48317296],
                   [0.01663708, -0.05670698]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dW2'],
            array([[-0.39202432, -0.13325855, -0.04601089]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['db2'],
            array([[0.15187861]]),
            rtol=0, atol=0.0001, equal_nan=False)

    def test_L_model_backward2(self):
        AL = array([[-0.00588594, -0.00873882]])
        Y_assess = array([[0, 1]])

        A0 = array([[0.02971382, -2.24825777],
                    [-0.26776186, 1.01318344],
                    [0.85279784, 1.1081875],
                    [1.11939066, 1.48754313]])
        W1 = array([[-1.11830068, 0.84583341, -1.86088953, -0.6028851],
                    [-1.91447204, 1.04814751, 1.33373782, -0.19741468],
                    [1.77464503, -0.67472751, 0.15061687, 0.1529457]])
        b1 = array([[-1.06419527], [0.43794661], [1.93897846]])
        Z1 = array([[-1.02493087, 0.89933845],
                    [-0.15450685, 1.7696273],
                    [0.48378835, 0.6762164]])

        A1 = array([[0.64316328, 0.24908671],
                    [-1.3957635, 1.39166291],
                    [-1.37066901, 0.23856319]])
        W2 = array([[0.61407709, -0.83791227, 0.14506321]])
        b2 = array([[1.16788229]])
        Z2 = array([[-0.02410447, -0.88865742]])

        grads = L_model_backward(AL, Y_assess,
                                 [((A0, W1, b1), Z1), ((A1, W2, b2), Z2)])

        assert_allclose(
            grads['dA0'],
            array([[6.39730387e-02, 2.77598463e+01],
                   [-2.43228186e-02, -1.07915269e+01],
                   [5.42949062e-03, -5.28896859e+01],
                   [5.51344139e-03, -4.31481965e+00]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dW1'],
            array([[-16.31042648, 7.35033779, 8.03956335, 10.79167311],
                   [22.25568547, -10.02958483, -10.97003769, -14.72530977],
                   [-3.85247073, 1.73154152, 1.9145542, 2.56948923]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['db1'],
            array([[7.25469593], [-9.89908088], [1.73179844]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dA1'],
            array([[0.15259879, 14.50939187],
                   [-0.20822206, -19.79816175],
                   [0.03604836, 3.42754853]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['dW2'],
            array([[3.02261935, 16.26765683, 2.64807504]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            grads['db2'],
            array([[11.93823295]]),
            rtol=0, atol=0.0001, equal_nan=False)

    def test_update_parameters(self):
        parameters = {}
        parameters['W1'] = array([
            [-0.41675785, -0.05626683, -2.1361961, 1.64027081],
            [-1.79343559, -0.84174737, 0.50288142, -1.24528809],
            [-1.05795222, -0.90900761, 0.55145404, 2.29220801]])
        parameters['W2'] = array([[-0.5961597, -0.0191305, 1.17500122]])
        parameters['W3'] = array([[-1.02378514, -0.7129932, 0.62524497],
                                  [-0.16051336, -0.76883635, -0.23003072]])
        parameters['b1'] = array([[0.04153939], [-1.11792545], [0.53905832]])
        parameters['b2'] = array([[-0.74787095]])

        grads = {}
        grads['dW1'] = array([[1.78862847, 0.43650985,
                               0.09649747, -1.8634927],
                              [-0.2773882, -0.35475898,
                               -0.08274148, -0.62700068],
                              [-0.04381817, -0.47721803,
                               -1.31386475, 0.88462238]])
        grads['dW2'] = array([[-0.40467741, -0.54535995, -1.54647732]])
        grads['db1'] = array([[0.88131804], [1.70957306], [0.05003364]])
        grads['db2'] = array([[0.98236743]])

        parameters = update_parameters(parameters, grads, 0.1)

        assert_allclose(
            parameters["W1"],
            array([[-0.59562069, -0.09991781, -2.14584584, 1.82662008],
                   [-1.76569676, -0.80627147, 0.51115557, -1.18258802],
                   [-1.0535704, -0.86128581, 0.68284052, 2.20374577]]),
            rtol=0, atol=0.0001, equal_nan=False)

        assert_allclose(
            parameters["W2"],
            array([[-0.55569196, 0.0354055, 1.32964895]]),
            rtol=0, atol=0.0001, equal_nan=False)
