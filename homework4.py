import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tqdm

# Load mnist from tensorflow_datasets
train_ds, test_ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True)

# Create our dataset
def prepare_math_minst_data(math_mnist, batch_size, subtask):
    # flatten the images into vector
    math_mnist = math_mnist.map(lambda img_x, target: (tf.reshape(img_x, (-1,)), target))
    # Convert data from uint8 to float32
    math_mnist = math_mnist.map(lambda img_x, target: (tf.cast(img_x, tf.float32), target))
    # Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    math_mnist = math_mnist.map(lambda img_x, target: ((img_x/128.)-1., target))
    # Shuffle data
    # we want to have two mnist images in each example
    # this leads to a single example being ((img_x1,img_y1),(img_x2,img_y2))
    zipped_dataset = tf.data.Dataset.zip((math_mnist.shuffle(2000),
                                        math_mnist.shuffle(2000)))
    # Subtask 1: a + b ≥ 5
    if subtask == 1:
        # Create targets
        zipped_dataset = zipped_dataset.map(lambda img_x1, img_x2: (img_x1[0], img_x2[0],
                                        tf.cast((img_x1[1] + img_x2[1] >= 5), tf.int32)))
    # Subtask 2: y == a - b 
    elif subtask == 2: 
        # Create targets
        zipped_dataset = zipped_dataset.map(lambda img_x1, img_x2: (img_x1[0], img_x2[0],
                                        tf.cast((img_x1[1] - img_x2[1]), tf.int32)))
        zipped_dataset = zipped_dataset.map(lambda img_x1, img_x2, target: (img_x1, img_x2, 
                                        tf.one_hot(target, depth=19)))
    # Cache this progress in memory
    zipped_dataset.cache()
    # Shuffle data
    zipped_dataset = zipped_dataset.shuffle(2000)
    # Batch data
    zipped_dataset = zipped_dataset.batch(batch_size)
    # Prefetch data
    zipped_dataset = zipped_dataset.prefetch(tf.data.AUTOTUNE)
    # Return preprocessed dataset
    return zipped_dataset

class MyNNmodel(tf.keras.Model):
    def __init__(self, optimizer, subtask):
        # Inherit functionality from parent class
        super(MyNNmodel, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        # Optimizer, loss function and metrix
        self.subtask = subtask
        self.optimizer = optimizer
        # layers to encode the images (both layers used for both images)
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)

        if self.subtask == 1:
            # loss function
            self.loss_function = tf.keras.losses.BinaryCrossentropy()
            # Metrix
            self.metrics_list = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'), 
                tf.keras.metrics.Mean(name='loss')
                ]
            self.out_layer = tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)

        elif self.subtask == 2:
            # loss function
            self.loss_function = tf.keras.losses.CategoricalCrossentropy()
            # Metrix
            self.metrics_list = [
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'), 
                tf.keras.metrics.Mean(name='loss')
                ]
            self.out_layer = tf.keras.layers.Dense(19,activation=tf.nn.softmax)

    # Call method(forward computation) 
    @tf.function
    def __call__(self, images, training=False):

        img1, img2 = images

        img1_x = self.flatten(img1)
        img2_x = self.flatten(img2)

        img1_x = self.dense1(img1)
        img1_x = self.dense2(img1_x)
        
        img2_x = self.dense1(img2)
        img2_x = self.dense2(img2_x)

        combined_x = tf.concat([img1_x, img2_x], axis=1)
        combined_x = self.dense3(combined_x)

        return self.out_layer(combined_x)

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
        img1, img2, label = data

        with tf.GradientTape() as tape:
            output = self((img1, img2), training=True)
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
        img1, img2, label = data

        # The same as train step(without parameter updates)
        output = self((img1, img2), training=True)
        loss = self.loss_function(label, output)

        # Update the state of the metrics according to loss
        self.metrics[0].update_state(label, output)
        self.metrics[1].update_state(loss)

        # Return a dictionary with metric names as keys and metric results as values
        return {m.name: m.result() for m in self.metrics}

def create_summary_writers(config_name):
    # Define where to save the logs
    # along with this, you may want to save a config file with the same name so you know what the hyperparameters were used
    # Alternatively make a copy of the code that is used for later reference
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_path = f"logs/{config_name}/{current_time}/train"
    test_log_path = f"logs/{config_name}/{current_time}/train"

    # Log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    # log writer for validation metrics
    test_summary_writer = tf.summary.create_file_writer(test_log_path)
    
    return train_summary_writer, test_summary_writer


def train_loop(model, train_ds, test_ds, start_epoch, 
                epochs, train_summary_writer, 
                test_summary_writer, save_path):

    #1. Iterate over epochs
    for epoch in range(start_epoch, epochs):
        # 2. Train steps on all batches in the training data
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
        
        # 3.log and print training metrics 
        with train_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                tf.summary.scalar(f'{metric.name}', metric.result(), step=epoch)
            # Alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)
            # e.g. tf.summary.image(name="mean_activation_layer3", data = metrics["mean_activation_layer3"],step=e)
        
        # Print the Epoch number
        print(f'Epoch: {epoch}')
        # Print the metrics
        print([f'{key}: {value.numpy()}' for (key, value) in metrics.items()])

        # 4. Reset metric objects
        model.reset_metrics()

        # 5. Evaluate on validation data
        for data in test_ds:
            metrics = model.test_step(data)

        # 6. Log validation metric
        with test_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                tf.summary.scalar(f'{metric.name}', metric.result(), step=epoch)
            # Alternatively, log metrics individually (allows for non-scalar metrics such as tf.keras.metrics.MeanTensor)

        print([f'Test_{key}: {value.numpy()}' for (key, value) in metrics.items()])
        # 7. Reset metric objects
        model.reset_metrics()

    # 8. Save the model weghts if save_path is given
    if save_path:
        model.save_weights(save_path)

def train(optimizer, save_path, batch_size, subtask):
    tf.keras.backend.clear_session()
    train_summary_writer, test_summary_writer = create_summary_writers(config_name='RUN1')

    train_dataset = prepare_math_minst_data(train_ds, batch_size, subtask)
    test_dataset = prepare_math_minst_data(test_ds, batch_size, subtask)

    # Сheck the contents of the dataset
    for img1, img2, label in train_dataset:
        print(img1.shape, img2.shape, label.shape)
        break

    print('\n\n')
    # Instance model with Adam optimizer
    model = MyNNmodel(optimizer, subtask)

    # Pass arguments to training loop function
    train_loop(model=model,
                train_ds=train_dataset,
                test_ds=test_dataset,
                start_epoch=0,
                epochs=10,
                train_summary_writer=train_summary_writer,
                test_summary_writer=test_summary_writer,
                save_path=save_path)


def main():
    # Instance optimizers
    optimizer_Adam = tf.keras.optimizers.Adam(learning_rate=0.01)
    optimizer_SGD = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    optimizer_SGD_no_momentum = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0) 
    optimizer_RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.01)
    optimizer_AdaGrad = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    # Choose a path to save the weights
    save_path = "../Homework4/"

    subtask_number = np.array([1,2])
    batch_size = 32

    optimizers_list = [
        optimizer_Adam, 
        optimizer_SGD, 
        optimizer_SGD_no_momentum,
        optimizer_RMSprop,
        optimizer_AdaGrad
    ]

    for optimizer in optimizers_list:
        # Train model for the first subtask
        train(optimizer, save_path, batch_size, subtask_number[0])
        # Train model for the second subtask
        train(optimizer, save_path, batch_size, subtask_number[1])

    
if __name__ == '__main__':
    main()    
