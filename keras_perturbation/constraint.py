import keras
import keras.backend as K


class Clip(keras.constraints.Constraint):

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def get_config(self):
        return {
            'min_value': self.min_value,
            'max_value': self.max_value,
        }

    def __call__(self, w):
        return K.clip(w, min_value=self.min_value, max_value=self.max_value)
