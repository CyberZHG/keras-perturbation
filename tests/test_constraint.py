import unittest
import keras
import keras.backend as K
from keras_perturbation import Clip


class TestConstraint(unittest.TestCase):

    def test_clip(self):
        weight = K.variable(value=keras.initializers.glorot_normal()((3, 5)),
                            dtype=K.floatx(),
                            name='Test')
        with K.get_session().as_default():
            values = Clip(min_value=-1.2, max_value=1.7)(weight - 10.0).eval()
            for row in values:
                for v in row:
                    self.assertAlmostEqual(-1.2, v)
            values = Clip(min_value=-1.2, max_value=1.7)(weight + 10.0).eval()
            for row in values:
                for v in row:
                    self.assertAlmostEqual(1.7, v)
