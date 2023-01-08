import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from tensorflow import keras
from keras import datasets, layers, models
import datetime
import tqdm

# Load cifar10 dataset
train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
train_images, train_labels = [], []
test_images, test_labels = [], []

for images, labels in train_ds:
    train_images.append(images)
    train_labels.append(labels)

for images, labels in test_ds:
    test_images.append(images)
    test_labels.append(labels)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']
#cifar10 = tf.keras.datasets.cifar10
#(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

"""
def prepare_cifar10_dataset(images, labels):
    
    labels = labels.flatten()

    # Reshape data
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 3)
    # Normalize data, just bringing image values from range [0, 255] to [0, 1]
    images = images / 255

    # Convert labels into one_hot encodings
    labels = tf.one_hot(labels.astype(np.int32), depth=10)

    return images, labels

"""
# Create a data pipeline function that prepares data for use in the model
def prepare_cifar10_dataset(cifar10):   
    # Convert data from uint8 to float32
    cifar10 = cifar10.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # Sloppy input normalization, just bringing image values from range [0, 255] to [0, 1]
    cifar10 = cifar10.map(lambda img, target: ((img/128.), target))
    # Create one-hot targets
    cifar10 = cifar10.map(lambda img, target: (img, tf.one_hot(tf.cast(target, tf.int32), depth=10)))
    # Cache this progress in memory, as there is no need to redo it; it is deterministic after all
    cifar10 = cifar10.cache()
    # Shuffle data
    cifar10 = cifar10.shuffle(1000)
    # Batch data
    cifar10 = cifar10.batch(32)
    # Prefetch data
    cifar10 = cifar10.prefetch(20)
    # Return preprocessed dataset
    return cifar10

# Visualize sample images
def plot_the_images(train_images, train_labels, names):
    plt.figure(figsize=(10,10))
    for item in range(20):
        plt.subplot(5, 5, item+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[item])
        # The CIFAR labels happen to be arrays, 
        # which is why you need the extra index
        plt.xlabel(names[train_labels[item][0]])
    plt.show()

# Create basic convolutional neural netwotk class
class BasicCNN(tf.keras.Model):
    def __init__(self, optimizer, num_classes, input_layer_shape):
        super(BasicCNN, self).__init__()

        self.optimizer = optimizer
        self.num_classes = num_classes
        self.input_layer_shape = input_layer_shape
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.metrics_list = [
            tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
            tf.keras.metrics.Mean(name='loss')    
        ] 

        self.dropout1 = tf.keras.layers.Dropout(0.25, input_shape=self.input_layer_shape[1:])

        self.conv_layer1 = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')
        self.conv_layer2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.pooling = tf.keras.layers.MaxPooling2D((2,2))
        # Prevents overfitting
        self.dropout2 = tf.keras.layers.Dropout(0.25)

        self.conv_layer3 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
        self.conv_layer4 = tf.keras.layers.Conv2D(64, 3, activation='relu')
        self.avg_global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout3 = tf.keras.layers.Dropout(0.25)

        self.flatten = tf.keras.layers.Flatten()
        self.out_layer = tf.keras.layers.Dense(self.num_classes, activation='softmax')

    @tf.function
    def __call__(self, x, training=False):
        if training:
            x = self.dropout1(x, training=training)
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.pooling(x)
        if training:
            x = self.dropout2(x, training=training)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.avg_global_pool(x)
        if training:
            x = self.dropout3(x, training=training)
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

def create_summary_writers(optimizer_name):
    # Define where to save the logs
    # along with this, you may want to save a config file with the same name so you know what the hyperparameters were used
    # Alternatively make a copy of the code that is used for later reference
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_path = f"logs/{optimizer_name}/{current_time}/train"
    test_log_path = f"logs/{optimizer_name}/{current_time}/test"

    # Log writer for training metrics
    train_summary_writer = tf.summary.create_file_writer(train_log_path)
    # Log writer for validation metrics
    test_summary_writer = tf.summary.create_file_writer(test_log_path)

    return train_summary_writer, test_summary_writer


def train_loop(model, train_ds, test_ds, start_epoch, 
                epochs, train_summary_writer, 
                test_summary_writer, save_path):

    # 1. Iterate over epochs
    for epoch in range(start_epoch, epochs):
        # 2. Train steps on all batches in the training data
        for data in tqdm.tqdm(train_ds, position=0, leave=True):
            metrics = model.train_step(data)
            
        # 3.log and print training metrics 
        with train_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                tf.summary.scalar(f'{metric.name}', metric.result(), step=epoch)

        # Print the epoch number
        print(f'Epoch: {epoch}')
        # Print the metrics
        print([f'{key}: {value.numpy()}' for (key, value) in metrics.items()])

        # 4. Reset metric objects
        model.reset_metrics()

        #5. Evaluate on validation data
        for data in test_ds:
            metrics = model.test_step(data)
            
        # 6. Log validation metric
        with test_summary_writer.as_default():
            # for scalar metrics:
            for metric in model.metrics:
                tf.summary.scalar(f'{metric.name}', metric.result(), step=epoch)

        print([f'Test_{key}: {value.numpy()}' for (key, value) in metrics.items()])
        # 7. Reset metric objects
        model.reset_metrics()

    # 8. Save the model weghts if save_path is given
    if save_path:
        model.save_weights(save_path)

def train(optimizer, save_path, class_names, num_classes, input_shape, optimizer_name): 
    train_summary_writer, test_summary_writer = create_summary_writers(optimizer_name)

    train_dataset = prepare_cifar10_dataset(train_ds,'train') 
    test_dataset = prepare_cifar10_dataset(test_ds, 'test')

    # Сheck the contents of the dataset
    for image, label in tfds.as_numpy(train_dataset):
        print(image.shape, label.shape)
        break
    print('\n\n')
    # Visualize sample images
    plot_the_images(train_images, train_labels, class_names)

    print('\n\n')

    model = BasicCNN(optimizer, num_classes, input_shape)

    train_loop(model=model,
                train_ds=train_dataset,
                test_ds=test_dataset,
                start_epoch=0,
                epochs=15,
                train_summary_writer=train_summary_writer,
                test_summary_writer=test_summary_writer,
                save_path=save_path)

def main():

    input_shape = (32,32,3)
    num_classes = 10

    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    # Choose a path to save the weights
    save_path = "../Homework5/"

    optimizer = tf.keras.optimizers.Adam()
    optimizer_name = 'Adam'

    train(optimizer, save_path, class_names, num_classes, input_shape, optimizer_name)

if __name__ == '__main__':
    main()