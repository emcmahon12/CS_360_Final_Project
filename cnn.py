"""
Convolutional neural network architecture. Copied from Lab 7
Author: Elle McMahon, Micah Painter
Date: 4/4/2024
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import fc_nn

##################

class CNNmodel(Model):
    """
    A convolutional neural network; the architecture is:
    Conv -> ReLU -> Conv -> ReLU -> Dense
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(CNNmodel, self).__init__()
        # Setting up the first layer with 32 filters, where each filter needs to be 5x5
        self.con1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), activation = 'relu', use_bias=True)
        # Setting up the second layer with 16 filters, where each filiter should be 3x3
        self.con2 = tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), activation = 'relu', use_bias=True)
        # # Third layer 
        # self.con3 = tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), activation = 'relu', use_bias=True)
        # # Forth layer
        # self.con4 = tf.keras.layers.Conv2D(filters=16, kernel_size = (3,3), activation = 'relu', use_bias=True)
        # Fifth layer is a fully connected neural network, called from fc_nn
        self.model =fc_nn.FCmodel()
        



    def call(self, x):
        '''Call method used to call / create the layers for a given x'''
        x = self.con1(x)
        x= self.con2(x)
        # x = self.con3(x)
        # x = self.con4(x)
        x= self.model.call(x)

        return x

def three_layer_convnet_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    cnn_model = CNNmodel()

    # shape is: number of examples (mini-batch size), width, height, depth
    x_np = np.random.rand(64, 32, 32, 3) # change here when changing size of images

    # call the model on this input and print the result
    output = cnn_model.call(x_np)
    print(output) 

    # Printing the shapes to make sure they make sense
    for v in cnn_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test three layer function
    three_layer_convnet_test()

if __name__ == "__main__":
    main()
