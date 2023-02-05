import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import datetime
import glob
import imageio
import PIL

from tensorflow import keras
from keras import datasets, layers
from keras import models, losses
from keras import Sequential

from cvae import * 

# Load MNIST dataset for convolutional autoencoder
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
# Prepare MNIST dataset for convolutional autoencoder
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
# x_train = tf.expand_dims(x_train, axis=0)
# x_test = tf.expand_dims(x_test, axis=0)

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print(x_train.shape)
print(x_test.shape)

noise_factor = 0.2
# Add a random noise to the images
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)

x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

# Prepare MNIST dataset for convolutional autoencoder
"""
def prepare_mnist_dataset(mnist_ds, noise_factor):
    # convert data from uint8 to float32
    mnist_ds = mnist_ds.map(lambda img, target: (tf.cast(img, dtype=tf.float32), target))
    # Sloppy input normalization, just bringing image values from range [0, 255] to [0, 1]
    mnist_ds = mnist_ds.map(lambda img, target: ((img/128.), target))
    # Normalize the images 
    mnist_ds = mnist_ds.map(lambda img, target: (tf.expand_dims(img, axis=-1), target))
    # Add a random noise to the images
    img_noisy = list(mnist_ds.map(lambda img, target: (img + noise_factor * tf.random.normal(shape=img.shape), 
                    target)))
    img_noisy = list(mnist_ds.map(lambda img, target: (tf.clip_by_value(img, clip_value_min=0., clip_value_max=1.), 
                    target)))
    # shuffle, batch, prefetch
    mnist_ds = mnist_ds.shuffle(1000)
    mnist_ds = mnist_ds.batch(32)
    mnist_ds = mnist_ds.prefetch(tf.data.AUTOTUNE)
    # return preprocessed dataset
    return mnist_ds, img_noisy
"""

def plot_noisy_images(x_test_noisy):
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i+1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x_test_noisy[i]), cmap=plt.cm.binary)

    plt.show()

# Create a convolutional autoencoder class
class Denoise(tf.keras.Model):
    def __init__(self):
        super(Denoise, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=16, kernel_size=(3, 3), strides=2, activation='relu', 
                padding='same', name='hidden_layer_1'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(
                filters=8, kernel_size=(3, 3),strides=2, activation='relu', 
                padding='same', name='hidden_layer_2'),
            # tf.keras.layers.GlobalAveragePooling2D(),
            # tf.keras.layers.Flatten(), 
            # Latent vector
            tf.keras.layers.Dense(units=4, activation='relu', name='code_layer')
        ])

        self.decoder = tf.keras.Sequential([
            # tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=8, kernel_size=3, strides=2, activation='relu', 
                padding='same', name='hidden_layer_3'),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=3, strides=2, activation='relu', 
                padding='same', name='hidden_layer_4'),
            tf.keras.layers.Conv2D(
                filters=1, kernel_size=(3, 3), activation='sigmoid',
                padding='same', name='output_layer')
        ])

    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

def visualize_conv_autoencoder_data(history):
    # Plotting the loss data
    plt.figure(figsize=(12,12))
    plt.plot(history.history['loss'], label='training')
    plt.plot(history.history['val_loss'], label='validation')
    plt.legend(loc='best')
    plt.xlabel('Epoch')
    plt.ylabel('Categorical Crossentropy loss')

    plt.show()

def plot_result_images(decoded_images, x_test_noisy):
    n = 10
    plt.figure(figsize=(20,4))
    for i in range(n):
        # display original + noise
        ax = plt.subplot(2, n, i+1)
        plt.title("original + noise")
        plt.imshow(tf.squeeze(x_test_noisy[i]))
        plt.get_xaxis().set_visible(False)
        plt.get_yaxis().set_visible(False)

        # display reconstruction
        bx = plt.subplot(2, n, i+n+1)
        plt.title("reconstructed")
        plt.imshow(tf.squeeze(decoded_images[i]))
        plt.get_xaxis().set_visible(False)
        plt.get_yaxis().set_visible(False)

        bx = plt.subplot(2, n, i+1)

    plt.show()

def train_conv_autoencoder(autoencoder, x_train, x_test, x_train_noisy_img, x_test_noisy_img):
    autoencoder.compile(
        optimizer='adam', 
        loss=losses.MeanSquaredError()
        )

    EXPERIMENT_NAME = 'RNN_cumsum'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f'./logs/{EXPERIMENT_NAME}/{current_time}')

    # Train convolutional autoencoder
    history = autoencoder.fit(x_train_noisy_img,
                                x_train,
                                batch_size=32,
                                epochs=10,
                                shuffle=True,
                                validation_data=(x_test_noisy_img, x_test),
                                callbacks=logging_callback)

    # Evaluate the model on the test data
    evaluate_res = autoencoder.evaluate

    # Save model
    autoencoder.save(filepath='../Homework8/saved_models/')

    visualize_conv_autoencoder_data(history)

def main():
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 2

    plot_noisy_images(x_test_noisy)

    # Define convolutional autoencoder object 
    conv_autoencoder_model = Denoise()
    # Define convolutional variational autoencoder object 
    cvae_model = CVAE(latent_dim)

    train_conv_autoencoder(conv_autoencoder_model, x_train, x_test, x_train_noisy, x_test_noisy)

    # Let's take a look at a summary of the encoder. 
    # Notice how the images are downsampled from 28x28 to 7x7
    conv_autoencoder_model.encoder.summary()
    # The decoder upsamples the images back from 7x7 to 28x28.
    conv_autoencoder_model.decoder.summary()

    # Plotting both the noisy images and the denoised images produced by the autoencoder.
    encoded_images = conv_autoencoder_model.encoder(x_test_noisy).numpy()
    decoded_images = conv_autoencoder_model.decoder(encoded_images).numpy()

    plot_result_images(decoded_images, x_test_noisy)

    train_CVAE(cvae_model, latent_dim)

if __name__ == '__main__':
    main()
