"""Some special pupropse layers for SSD."""
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import numpy as np


class Normalize(Layer):
    """Normalization layer as described in ParseNet paper.
    # Arguments
        scale: Default feature scale.
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)
    # Output shape
        Same as input
    # References
        http://cs.unc.edu/~wliu/papers/parsenet.pdf
    # TODO
        Add possibility to have one scale for all features.
    """
    def __init__(self, scale=20, **kwargs):
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name=self.name+'_gamma', 
                                     shape=(input_shape[-1],),
                                     initializer=initializers.Constant(self.scale), 
                                     trainable=True)
        super(Normalize, self).build(input_shape)
        
    def call(self, x, mask=None):
        return self.gamma * K.l2_normalize(x, axis=-1)

