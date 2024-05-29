"""
Fully connected neural network architecture. Copied from Lab 7
Author: Micah Painter, Elle McMahon
Date: 04/04/2024
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

##################

class FCmodel(Model):
    """
    A fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(FCmodel, self).__init__()

        # flatten the data so that we can put it in a fully connected network
        self.flatten=tf.keras.layers.Flatten()
    
        # change the number of classes here if runing the neural network for a subset labels
        self.d1= tf.keras.layers.Dense(units=4000, activation= tf.keras.activations.relu)
        self.d2 = tf.keras.layers.Dense(units = 24, activation = tf.keras.activations.softmax)
        self.classes = 24

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x= self.d2(x)
        return x

def two_layer_fc_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    fc_model = FCmodel()

    # shape is: number of examples (mini-batch size), width, height, depth
    #x_np = np.zeros((64, 32, 32, 3))
    x_np = np.random.rand(64, 32, 32, 3) # change here when changing size of images

    # call the model on this input and print the result
    output = fc_model.call(x_np)
    print(output)

    # Print model parameter shapes to make sure they make sense
    for v in fc_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)
