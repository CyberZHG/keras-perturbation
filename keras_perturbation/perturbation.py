import keras
import keras.backend as K
from .constraint import Clip


class Perturbation(keras.layers.Layer):

    def __init__(self,
                 max_variable_shape,
                 constraint=Clip(min_value=-1e5, max_value=1e5),
                 **kwargs):
        self.variable_len = len(max_variable_shape)
        self.max_variable_shape = list(max_variable_shape)
        self.constraint = keras.constraints.get(constraint)
        self.supports_masking = True
        self.perturbation = None
        super(Perturbation, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'max_variable_shape': self.max_variable_shape,
            'constraint': self.constraint,
        }
        base_config = super(Perturbation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.perturbation = self.add_weight(shape=tuple(self.max_variable_shape) + input_shape[self.variable_len:],
                                            name='{}_W'.format(self.name),
                                            constraint=self.constraint,
                                            initializer=keras.initializers.get('zeros'),
                                            dtype=K.floatx())
        super(Perturbation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        current_shape = K.shape(inputs)[:self.variable_len]
        return inputs + self.get_perturbation(current_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return tuple([None] * self.variable_len) + input_shape[self.variable_len:]

    def get_perturbation(self, shape):
        key = tuple(slice(None, shape[i], None) for i in range(self.variable_len))
        return self.perturbation[key]

    def get_perturbation_values(self, shape):
        key = tuple(slice(None, x, None) for x in shape)
        return self.perturbation[key].eval(session=K.get_session())
