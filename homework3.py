import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], 
                                        as_supervised=True, with_info=True)

# Show the information about MNIST dataset
# print(ds_info)
# tfds.show_examples(train_ds, ds_info)

# Create a data pipeline function that prepares data for use in the model
def prepare_mnist_data(mnist):
    # flatten the images into vectors
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # Convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img/128.)-1., target))
    # Create one-hot targets
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # Cache this progress in memory, as there is no need to redo it; it is deterministic after all
    mnist = mnist.cache()
    # Shuffle data
    mnist = mnist.shuffle(1000)
    # Batch data
    mnist = mnist.batch(32)
    # Prefetch data
    mnist = mnist.prefetch(20)
    # Return preprocessed dataset
    return mnist

# Create feed-froward neural network class
class NNModel(tf.keras.Model):
    def __init__(self):
        super(NNModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)

        return x

# Train neural network
def train_step(model, input, target, loss_function, optimizer):
    # Loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input)
        loss = loss_function(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def test(model, test_data, loss_function):
    # Test over complete test data

    test_accuracy_aggregator = []
    test_loss_aggregator = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy

def visualization (train_losses, test_losses, test_accuracies):
    """ Visualizes accuracy and loss for training and test data using
    the mean of each epoch . 
    Loss is displayed in a regular line , accuracy in a dotted
    line . 
    Training data is displayed in blue , test data in red . 
    Parameters
    ----------
    train_losses : numpy . ndarray
      training losses
    train_accuracies : numpy . ndarray
      training accuracies
    test_losses : numpy . ndarray
      test losses
    test_accuracies : numpy . ndarray
      test accuracies
    """

    plt.figure()
    line1, = plt.plot(train_losses, 'b-')
    line2, = plt.plot(test_losses, 'r-')
    line3, = plt.plot(test_accuracies, 'g-')
    plt.xlabel('Training steps')
    plt.ylabel('Lose/Accuracy')
    plt.legend((line1,line2,line3), ('training','test','test accuracy'))

    plt.show()

def main():

    tf.config.set_soft_device_placement(True)
    tf.debugging.set_log_device_placement(True)

    tf.keras.backend.clear_session()

    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)

    # For showcasing we only use a subset of the training and test data (generally use all of the available data!)
    train_dataset = train_dataset.take(1000)
    test_dataset = test_dataset.take(100)

    ### Hyperparameters
    num_epochs = 10
    learning_rate = 0.001

    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer = tf.keras.optimizers.SGD(learning_rate)

    # Initialize lists for later visualization.
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Testing once before we begin
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    #check how model performs on train data once before we begin
    train_loss, _ = test(model, train_dataset, cross_entropy_loss)
    train_losses.append(train_loss)

    # We train for num_epochs epochs
    for epoch in range(num_epochs):
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

        # Training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_dataset:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)
    
        # Track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        #testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    visualization(train_losses, test_losses, test_accuracies)

if __name__ == '__main__':
    main()