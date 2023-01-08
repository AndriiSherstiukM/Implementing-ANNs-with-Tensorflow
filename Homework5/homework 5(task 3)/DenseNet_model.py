import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np

class DenselyConnectedCNNLayer(tf.keras.layers.Layer):
    def __init__(self, num_filters):
        super(DenselyConnectedCNNLayer, self).__init__()
        self.conv_layer = tf.keras.layers.Conv2D(num_filters, (3,3), 
                                                    padding='same', activation='relu')

    @tf.function
    def __call__(self, x):
        c = self.conv_layer(x)
        x = tf.concat((x,c), axis=-1)
    
        return x

class DenselyConnectedCNNBlock(tf.keras.layers.Layer):
    def __init__(self, num_filters, layers):
        super(DenselyConnectedCNNBlock, self).__init__()
        self.conv_layers = [DenselyConnectedCNNLayer(num_filters) for _ in range(layers)]

    @tf.function
    def __call__(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        return x

class DenselyConnectedCNN(tf.keras.Model):
    def __init__(self, optimizer, num_classes):
        super(DenselyConnectedCNN, self).__init__()

        self.optimizer = optimizer
        self.num_classes = num_classes
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.metrics_list = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Mean(name='loss')
        ]

        self.dence_block1 = DenselyConnectedCNNBlock(32, 4)
        self.pooling1 = tf.keras.layers.MaxPooling2D((2,2))

        self.dence_block2 = DenselyConnectedCNNBlock(64, 4)
        self.pooling2 = tf.keras.layers.MaxPooling2D((2,2))

        self.dence_block3 = DenselyConnectedCNNBlock(64, 4)
        self.avg_global_pool = tf.keras.layers.GlobalAvgPool2D()

        self.flatten = tf.keras.layers.Flatten()
        self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    @tf.function
    def  __call__(self, x, training=False):
        x = self.dence_block1(x)
        x = self.pooling1(x)
        x = self.dence_block2(x)
        x = self.pooling2(x)
        x = self.dence_block3(x)
        x = self.avg_global_pool(x)
        x = self.flatten(x)
        x = self.out_layer(x)

        return x

    # Metrics property
    @property
    def metrics(self):
        return self.metrics_list
        # return a list with all metrics in the model

    # Reset all metrics objects
    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_states()

    # Train step method
    @tf.function
    def train_step(self, data):
        image, label = data
        
        with tf.GradientTape() as tape:
            output = self(image, training=True)
            loss = self.loss_function(label, output)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update the state of the metrics according to loss
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)

        # Return a dictionary with metric names as keys and metric results as values
        return {m.name: m.result() for m in self.metrics}

    # Test step method
    @tf.function
    def test_step(self, data):
        image, label = data
        # The same as train step(without parameter updates)
        output = self(image, training=True)
        loss = self.loss_function(label, output)

        # Update the state of the metrics according to loss
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)

        # Return a dictionary with metric names as keys and metric results as values
        return {m.name: m.result() for m in self.metrics}
