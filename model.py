"""
turru.and.gors, 2020

During my experiments, I tried with the following structures:
    
   ------------------------------------------------------
    Layer       | Experiment 1  | Experiment 2
   ------------------------------------------------------
    conv 1      | 8             | 16
   ------------------------------------------------------
    conv 2      | 16            | 32
   ------------------------------------------------------
    conv 3      | 32            | 32
   ------------------------------------------------------
    dense       | 128           | 256
   ======================================================
    EPOCHS      | 100           | 200
   ======================================================
    ACCURACY    | 0.997         | 0.996
   ------------------------------------------------------
    VAL ACCURACY| 0.740         | 0.915
   ======================================================
   

"""

import tensorflow as tf


def create_model(input_shape, num_outputs):
    """
    Create keras model. 
    
    NOTE: using pure keras instead of tf.keras leads to errors when
    trying to feed the model with a tf.data.Dataset.
    https://github.com/tensorflow/tensorflow/issues/20698
    
    :param input_shape: Shape of the input image
    :type input_shape: Tuple
    
    :param num_outputs: Number of classes.
    :type num_outputs: integer
    
    :return: Neural network model
    :rtype: tf.keras.Model
    """
    inputs = tf.keras.Input(shape           = input_shape,
                         name               = "input")
    
    net = tf.keras.layers.Conv2D(filters    = 16, 
                                kernel_size = (3,3), 
                                padding     = "same", 
                                activation  = tf.nn.relu)(inputs)
    net = tf.keras.layers.MaxPooling2D()(net)
    
    net = tf.keras.layers.Conv2D(filters    = 32, 
                                kernel_size = (3,3), 
                                padding     = "same", 
                                activation  = tf.nn.relu)(net)
    net = tf.keras.layers.MaxPooling2D()(net)
    
    net = tf.keras.layers.Conv2D(filters    = 32, 
                                kernel_size = (3,3), 
                                padding     = "same", 
                                activation  = tf.nn.relu)(net)
    net = tf.keras.layers.MaxPooling2D()(net)
    
    net = tf.keras.layers.Flatten()(net)
    net = tf.keras.layers.Dense(units       = 256,
                               activation   = tf.nn.relu)(net)
    
    outputs = tf.keras.layers.Dense(units   = num_outputs)(net)
    
    model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model
    