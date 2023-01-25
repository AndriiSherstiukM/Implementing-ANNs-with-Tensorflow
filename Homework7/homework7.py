import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Load MNIST dataset
train_ds, test_ds = tfds.load('mnist', split=['train','test'], as_supervised=True)

def cumsum_dataset(ds, seq_len):
    # only get the targets, to keep this demonstration simple (and force students to understand the code if they are using it by rewriting it respectively)
    ds = ds.map(lambda x, t: tf.cast(t, dtype=tf.dtypes.int32))
    # use window to create subsequences. This means ds is not a dataset of datasets, i.e. every single entry in the dataset is itself a small tf.data.Dataset object with seq_len many entries!
    ds = ds.window(seq_len)
    # make sure to check tf.data.Dataset.scan() to understand how this works!
    def alternating_scan_function(state, elem):
        # state is allways the sign to use!
        old_sign = state
        # just flip the sign for every element
        new_sign = old_sign*-1
        # elem is just the target of the element. We need to apply the appropriate sign to it!
        signed_target = elem*old_sign
        # we need to return a tuple for the scan function: The new state and the output element
        out_elem = signed_target
        new_state = new_sign
        return new_state, out_elem
    # we now want to apply this function via scanning, resulting in a dataset where the signs are alternating
    # remember we have a dataset, where each element is a sub dataset due to the windowing!
    ds = ds.map(lambda sub_ds: sub_ds.scan(initial_state=1, scan_func=alternating_scan_function))
    # now we need a scanning function which implements a cumulative sum, very similar to the cumsum used above
    def scan_cum_sum_function(state, elem):
        # state is the sum up the the current element, element is the new digit to add to it
        sum_including_this_elem = state+elem
        # both the element at this position and the returned state should just be sum up to this element, saved in sum_including_this_elem
        return sum_including_this_elem, sum_including_this_elem
    # again we want to apply this to the subdatasets via scan, with a starting state of 0 (sum before summing is zero...)
    ds = ds.map(lambda sub_dataset: sub_dataset.scan(initial_state=0, scan_func=scan_cum_sum_function))
    # finally we need to create a single element from everything in the subdataset
    ds = ds.map(lambda sub_dataset: sub_dataset.batch(seq_len).get_single_element())
    return ds
    
def prepare_mnist_dataset(mnist_ds, seq_len, ds_type):
    # choose type of dataset
    if ds_type == 'Train':
        # flatten the images into vector
        mnist_ds = mnist_ds.map(lambda img, target: (tf.reshape(img, (-1,28,28,1)), target))
        # convert data from uint8 to float32
        mnist_ds = mnist_ds.map(lambda img, target: (tf.cast(img, tf.float32), target))
        # Sloppy input normalization, just bringing image values from range [0, 255] to [0, 1]
        mnist_ds = mnist_ds.map(lambda img, target: ((img/128.), target))
        mnist_ds.apply(lambda dataset: cumsum_dataset(dataset, seq_len)).take(10)
        # shuffle, batch, prefetch
        mnist_ds = mnist_ds.shuffle(1000)
        mnist_ds = mnist_ds.batch(32)
        mnist_ds = mnist_ds.prefetch(tf.data.AUTOTUNE)
        # return preprocessed dataset
        return mnist_ds
    elif ds_type == 'Test':
        # flatten the images into vector
        mnist_ds = mnist_ds.map(lambda img, target: (tf.reshape(img, (-1,28,28,1)), target))
        # convert data from uint8 to float32
        mnist_ds = mnist_ds.map(lambda img, target: (tf.cast(img, tf.float32), target))
        # Sloppy input normalization, just bringing image values from range [0, 255] to [0, 1]
        mnist_ds = mnist_ds.map(lambda img, target: ((img/128.), target))
        mnist_ds.apply(lambda dataset: cumsum_dataset(dataset, seq_len)).take(10)
        # batch, prefetch
        mnist_ds = mnist_ds.batch(32)
        mnist_ds = mnist_ds.prefetch(tf.data.AUTOTUNE)
        # return preprocessed dataset
        return mnist_ds

class RNNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, cnn, recurrent_units_1, recurrent_units_2):
        super().__init__()

        self.recurrent_units_1 = recurrent_units_1
        self.recurrent_units_2 = recurrent_units_2

        self.linear_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(recurrent_units_1))
        self.linear_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(recurrent_units_2))
    
        # First recurrent layer in the RNN
        self.recurrent_layer_1 = tf.keras.layers.Conv2D(filters=recurrent_units_1,
                                    kernel_size=3,
                                    padding = 'same',
                                    kernel_initializer=tf.keras.initializers.Orthogonal(
                                        gain=1.0, seed=None),
                                    activation=tf.nn.tanh)
        
        # layer normalization for trainability
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        # Second recurrent layer in the RNN
        self.recurrent_layer_2 = tf.keras.layers.Conv2D(filters=recurrent_units_2,
                                    kernel_size=3,
                                    padding = 'same',
                                    kernel_initializer=tf.keras.initializers.Orthogonal(
                                        gain=1.0, seed=None),
                                    activation=tf.nn.tanh)

        # layer normalization for trainability
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

    @property
    def state_size(self):
        return [tf.TensorShape([self.recurrent_units_1]), 
                tf.TensorShape([self.recurrent_units_2])]

    @property
    def output_size(self):
        return [tf.TensorShape([self.recurrent_units_2])]

    def initial_state(self, inputs=None, batch_size=None, dtype=None):
        return [tf.zeros([self.recurrent_units_1]),
                tf.zeros([self.recurrent_units_2])]
    
    @tf.function
    def __call__(self, inputs, states):
        # Unpack the states
        state_layer_1 = states[0]
        state_layer_2 = states[1]

        # Linearly project input
        x = self.linear_1(inputs) + state_layer_1
        # Apply first reccurent layer
        new_state_layer_1 = self.recurrent_layer_1(x)
        # Apply first layer's layer norm
        x = self.batch_norm_1(new_state_layer_1)
        # linearly project output of layer norm
        x = self.linear_2(x) + state_layer_2
        # Apply second reccurent layer
        new_state_layer_2 = self.recurrent_layer_2(x)
        # Apply second layer's layer norm
        x = self.batch_norm_2(new_state_layer_2)

        # Return output and the list of new states of the layers
        return x, [new_state_layer_1, new_state_layer_2] 

    def get_config(self):
        return {'recurrent_units_1': self.recurrent_units_1,
                'recurrent_units_2': self.recurrent_units_2}
        
class RNNModel(tf.keras.Model):
    def __init__(self, input_n, output_n):
        super().__init__()

        self.rnn_cell = RNNCell(input_n, output_n)
        # Return_sequences collects and returns the output of the rnn_cell for all time-steps
        # Unroll unrolls the network for speed (at the cost of memory)
        self.rnn_layer = tf.keras.layers.RNN(self.rnn_cell, return_sequences=False, unroll=True)

        self.avg_global_pool = tf.keras.layers.TimeDistributed(
            tf.keras.layers.GlobalAvgPool2D())

        self.flatten = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Flatten())

        self.output_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(units=10,activation=tf.nn.sigmoid))

        self.metrics_list = [
            tf.keras.metrics.Mean(name='loss'),
            tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        ]

    @tf.function
    def __call__(self, sequence, training=False):
        x = self.rnn_layer(sequence)
        x = self.avg_global_pool(sequence)
        x = self.flatten(sequence)
        return self.output_layer(x)

    @property
    def metrics(self):
        return self.metrics_list

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset_state()

    def train_step(self, data):
        sequence, label = data
        with tf.GradientTape() as tape:
            output = self(sequence, training=True)
            loss = self.compiled_loss(label, output, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)
        
        return {m.name : m.result() for m in self.metrics}

    def test_step(self, data):
        sequence, label = data
        output = self(sequence, training=False)
        loss = self.compiled_loss(label, output, regularization_losses=self.losses)
                
        self.metrics[0].update_state(loss)
        self.metrics[1].update_state(label, output)
        
        return {m.name : m.result() for m in self.metrics}      

def visualization(history):
    # Plotting the loss data
    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy loss')
    

def main():

    SEQUENCE_LENGTH = 10

    train_dataset = prepare_mnist_dataset(train_ds, SEQUENCE_LENGTH, 'Train')
    # Сheck the contents of the dataset
    for img, label in train_dataset:
        print(img.shape, label.shape)
        break

    val_dataset = prepare_mnist_dataset(train_ds, SEQUENCE_LENGTH, 'Test')
    for img, label in val_dataset:
        print(img.shape, label.shape)
        break


    input_units = 24
    output_units = 48

    model = RNNModel(input_units, output_units)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # Compile the model
    model.compile(optimizer=optimizer,
                    loss=loss) 

    model.summary()

    EXPERIMENT_NAME = 'RNN_cumsum'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{EXPERIMENT_NAME}/{current_time}')

    # Train the model with fit function
    history = model.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=20,
                        callbacks = logging_callback)
    
    # Save model
    model.save(filepath='./saved_models/')

    # Visualize the data
    visualization(history)

if __name__ == '__main__':
    main()