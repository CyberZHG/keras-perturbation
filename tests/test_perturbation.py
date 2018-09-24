import unittest
import os
import tempfile
import random
import keras
import numpy as np
from keras_perturbation import Perturbation, Clip


class TestPerturbation(unittest.TestCase):

    def test_add_perturbation(self):
        model = keras.models.Sequential()
        model.add(Perturbation(
            max_variable_shape=(8,),
            constraint=Clip(-0.5, 0.5),
            input_shape=(6,),
            name='Perturbation',
        ))
        model.build()
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.mean_absolute_error,
            metrics=[keras.metrics.mean_absolute_error],
        )
        data_x = np.asarray([[0.0] * 6] * 1000)
        data_y = np.asarray([[-1.0, 1.0] * 3] * 1000)
        model.fit(
            x=data_x,
            y=data_y,
            batch_size=7,
            epochs=10,
        )
        model_path = os.path.join(tempfile.gettempdir(), 'test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Clip': Clip,
            'Perturbation': Perturbation,
        })
        perturbation = model.layers[0]
        values = perturbation.get_perturbation_values(shape=(7,))
        self.assertEqual(values.shape, (7, 6))
        for row in values:
            for i, v in enumerate(row):
                if i % 2 == 0:
                    self.assertAlmostEqual(-0.5, v)
                else:
                    self.assertAlmostEqual(0.5, v)
