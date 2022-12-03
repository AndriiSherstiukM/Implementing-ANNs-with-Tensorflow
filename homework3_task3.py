import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train','test'], 
                                        as_supervised=True, with_info=True)

# Create a data pipeline function that prepares data for use in the model
# Test 1, 2
def prepare_mnist_data_12(mnist):
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
    mnist = mnist.batch(16)
    # Prefetch data
    mnist = mnist.prefetch(20)
    # Return preprocessed dataset
    return mnist

# Test 3, 4
def prepare_minst_data_34(mnist):
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

def visualization (train_losses, test_losses, test_accuracies, attempt_number):
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
    plt.title(f'Attempt {attempt_number}')

    plt.show()

def main():
    tf.keras.backend.clear_session()

    train_dataset_12 = train_ds.apply(prepare_mnist_data_12)
    test_dataset_12 = test_ds.apply(prepare_mnist_data_12)
    train_dataset_34 = train_ds.apply(prepare_minst_data_34)
    test_dataset_34 = test_ds.apply(prepare_minst_data_34)

    # For showcasing we only use a subset of the training and test data (generally use all of the available data!)
    train_dataset_12 = train_dataset_12.take(1000)
    test_dataset_12 = test_dataset_12.take(100)
    train_dataset_34 = train_dataset_34.take(1000)
    test_dataset_34 = test_dataset_34.take(100)

    ### Hyperparameters. 
    # Number of epochs = 25 on test 1 and 3
    # Number of epochs = 10 on test 2 and 4
    num_epochs_13 = 25
    num_epochs_24 = 10

    ### Hyperparameters.
    # Learning rate = 0.3 on test 1 and 4
    # Learning rate = 0.001 on test 2 and 3
    learning_rate_14 = 0.1
    learning_rate_23 = 0.001

    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer_14 = tf.keras.optimizers.SGD(learning_rate_14)
    optimizer_23 = tf.keras.optimizers.SGD(learning_rate_23)

    # Initialize lists for later visualization.
    train_losses_1 = []
    train_losses_2 = []
    train_losses_3 = []
    train_losses_4 = []
    
    test_losses_1 = []
    test_losses_2 = []
    test_losses_3 = []
    test_losses_4 = []
    
    test_accuracies_1 = []
    test_accuracies_2 = []
    test_accuracies_3 = []
    test_accuracies_4 = []

    # Test 1
    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer_14 = tf.keras.optimizers.SGD(learning_rate_14)
    # Testing once before we begin
    test_loss_1, test_accuracy_1 = test(model, test_dataset_12, cross_entropy_loss)
    test_losses_1.append(test_loss_1)
    test_accuracies_1.append(test_accuracy_1)

    #check how model performs on train data once before we begin
    train_loss_1, _ = test(model, train_dataset_12, cross_entropy_loss)
    train_losses_1.append(train_loss_1)

    # We train for num_epochs epochs
    for epoch_1 in range(num_epochs_13):
        print(f'Epoch: {str(epoch_1)} starting with accuracy {test_accuracies_1[-1]}')

        # Training (and checking in with training)
        epoch_loss_agg_1 = []
        for input, target in train_dataset_12:
            train_loss_1 = train_step(model, input, target, cross_entropy_loss, optimizer_14)
            epoch_loss_agg_1.append(train_loss_1)
    
        # Track training loss
        train_losses_1.append(tf.reduce_mean(epoch_loss_agg_1))

        #testing, so we can track accuracy and test loss
        test_loss_1, test_accuracy_1 = test(model, test_dataset_12, cross_entropy_loss)
        test_losses_1.append(test_loss_1)
        test_accuracies_1.append(test_accuracy_1)

    visualization(train_losses_1, test_losses_1, test_accuracies_1, 1)

    # Test 2
    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer_23 = tf.keras.optimizers.SGD(learning_rate_23)
    # Testing once before we begin
    test_loss_2, test_accuracy_2 = test(model, test_dataset_12, cross_entropy_loss)
    test_losses_2.append(test_loss_2)
    test_accuracies_2.append(test_accuracy_2)

    #check how model performs on train data once before we begin
    train_loss_2, _ = test(model, train_dataset_12, cross_entropy_loss)
    train_losses_2.append(train_loss_2)

    # We train for num_epochs epochs
    for epoch_2 in range(num_epochs_24):
        print(f'Epoch: {str(epoch_2)} starting with accuracy {test_accuracies_2[-1]}')

        # Training (and checking in with training)
        epoch_loss_agg_2 = []
        for input, target in train_dataset_12:
            train_loss_2 = train_step(model, input, target, cross_entropy_loss, optimizer_23)
            epoch_loss_agg_2.append(train_loss_2)
    
        # Track training loss
        train_losses_2.append(tf.reduce_mean(epoch_loss_agg_2))

        #testing, so we can track accuracy and test loss
        test_loss_2, test_accuracy_2 = test(model, test_dataset_12, cross_entropy_loss)
        test_losses_2.append(test_loss_2)
        test_accuracies_2.append(test_accuracy_2)

    visualization(train_losses_2, test_losses_2, test_accuracies_2, 2)

    # Test 3
    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer_23 = tf.keras.optimizers.SGD(learning_rate_23)
    
    # Testing once before we begin
    test_loss_3, test_accuracy_3 = test(model, test_dataset_34, cross_entropy_loss)
    test_losses_3.append(test_loss_3)
    test_accuracies_3.append(test_accuracy_3)

    #check how model performs on train data once before we begin
    train_loss_3, _ = test(model, train_dataset_34, cross_entropy_loss)
    train_losses_3.append(train_loss_3)

    # We train for num_epochs epochs
    for epoch_3 in range(num_epochs_13):
        print(f'Epoch: {str(epoch_3)} starting with accuracy {test_accuracies_3[-1]}')

        # Training (and checking in with training)
        epoch_loss_agg_3 = []
        for input, target in train_dataset_34:
            train_loss_3 = train_step(model, input, target, cross_entropy_loss, optimizer_23)
            epoch_loss_agg_3.append(train_loss_3)
    
        # Track training loss
        train_losses_3.append(tf.reduce_mean(epoch_loss_agg_3))

        #testing, so we can track accuracy and test loss
        test_loss_3, test_accuracy_3 = test(model, test_dataset_34, cross_entropy_loss)
        test_losses_3.append(test_loss_3)
        test_accuracies_3.append(test_accuracy_3)

    visualization(train_losses_3, test_losses_3, test_accuracies_3, 3)

    # Test 4
    # Initialize the model.
    model = NNModel()
    # Initialize the loss: categorical cross entropy. Check out 'tf.keras.losses'.
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
    # Initialize the optimizer: SGD with default parameters. Check out 'tf.keras.optimizers'
    optimizer_14 = tf.keras.optimizers.SGD(learning_rate_14)
    # Testing once before we begin
    test_loss_4, test_accuracy_4 = test(model, test_dataset_34, cross_entropy_loss)
    test_losses_4.append(test_loss_4)
    test_accuracies_4.append(test_accuracy_4)

    #check how model performs on train data once before we begin
    train_loss_4, _ = test(model, train_dataset_34, cross_entropy_loss)
    train_losses_4.append(train_loss_4)

    # We train for num_epochs epochs
    for epoch_4 in range(num_epochs_24):
        print(f'Epoch: {str(epoch_4)} starting with accuracy {test_accuracies_4[-1]}')

        # Training (and checking in with training)
        epoch_loss_agg_4 = []
        for input, target in train_dataset_34:
            train_loss_4 = train_step(model, input, target, cross_entropy_loss, optimizer_14)
            epoch_loss_agg_4.append(train_loss_4)
    
        # Track training loss
        train_losses_4.append(tf.reduce_mean(epoch_loss_agg_4))

        #testing, so we can track accuracy and test loss
        test_loss_4, test_accuracy_4 = test(model, test_dataset_34, cross_entropy_loss)
        test_losses_4.append(test_loss_4)
        test_accuracies_4.append(test_accuracy_4)

    visualization(train_losses_4, test_losses_4, test_accuracies_4, 4)

if __name__ == '__main__':
    main()
