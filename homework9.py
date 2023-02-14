import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import PIL
import time
import urllib
import tqdm
import datetime

from tensorflow import keras
from keras import datasets, layers
from keras import models, losses
from IPython import display
from keras.utils import Progbar

categories = [
    line.rstrip(b'\n') for line in urllib.request.urlopen(
    'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')
    ]
print(categories[:10])

category = 'candle'

# Creates a folder to download the original drawings into.
# We chose to use the numpy format : 1x784 pixel vectors, with values going from 0 (white) to 255 (black). 
# We reshape them later to 28x28 grids and normalize the pixel intensity to [-1, 1]
if not os.path.isdir('../npy_files_homework9'):
    os.mkdir('../npy_files_homework9/')

url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'  
urllib.request.urlretrieve(url, f'../npy_files_homework9/{category}.npy')

images = np.load(f'../npy_files_homework9/{category}.npy')
print(f'{len(images)} images to train on')

# You can limit the amount of images you use for training by setting :
train_images = images[:10000]

# Prepare a dataset
def prepare_dataset(images, batch_size, buffer_size):
    images = images.reshape(train_images.shape[0], 28, 28, 1)
    # Normalize the images to [-1, 1]
    images = (images - 127.5) / 127.5
    dataset = tf.data.Dataset.from_tensor_slices(images)
    # Batch, shuffle and prefetch the data
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# Create discriminator class
class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu', 
            padding='same', input_shape=[28, 28, 1])
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        # layer normalization for trainability
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()

        self.conv_layer_2 = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same')
        self.dropout_2 = tf.keras.layers.Dropout(0.3)
        # layer normalization for trainability
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.out_layer = tf.keras.layers.Dense(
            units=1, activation='sigmoid'
        )

    def call(self, inputs, training=False):
        x = self.conv_layer_1(inputs)
        if training:
            x = self.dropout_1(x, training=training)
        x = self.batch_norm_1(x)
        x = self.conv_layer_2(x)
        if training:
            x = self.dropout_2(x, training=training)
        x = self.batch_norm_2(x)
        x = self.flatten(x)
        x = self.out_layer(x)

        return x

# Create generator class
class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_layer = tf.keras.layers.Dense(
            7*7*128, use_bias=False, input_shape=(100,))
        # layer normalization for trainability
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.leaky_relu_1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.reshape = tf.keras.layers.Reshape((7,7,128))

        self.conv_transpose_1 = tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=(3, 3), strides=(1, 1), 
            use_bias=False, padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.leaky_relu_2 = tf.keras.layers.LeakyReLU(alpha=0.3)

        self.conv_transpose_2 = tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=(3, 3), strides=(2, 2), 
            use_bias=False, padding='same')   
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()
        self.leaky_relu_3 = tf.keras.layers.LeakyReLU(alpha=0.3)

        self.out_layer = tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=(3, 3), strides=(2, 2), activation='tanh',
            use_bias=False, padding='same')

    def call(self, inputs, training=False):
        x = self.latent_layer(inputs)
        x = self.batch_norm_1(x)
        x = self.leaky_relu_1(x)
        x = self.reshape(x)
        x = self.conv_transpose_1(x)
        x = self.batch_norm_2(x)
        x = self.leaky_relu_2(x)
        x = self.conv_transpose_2(x)
        x = self.batch_norm_3(x)
        x = self.leaky_relu_3(x)
        x = self.out_layer(x)

        return x
        
# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Quantifies how well the discriminator is able to distinguish real images from fakes. 
# It compares the discriminator's predictions on real images to an array of 1s, and the 
# discriminator's predictions on fake (generated) images to an array of 0s.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, 
# if the generator is performing well, the discriminator will classify the fake images as real (or 1). 
# Here, compare the discriminators decisions on the generated images to an array of 1s.
def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    return loss

def saves_checkpoints(generator, discriminator, generator_optimizer, discriminator_optimizer):
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                        discriminator_optimizer=discriminator_optimizer,
                                        generator=generator,
                                        discriminator=discriminator)

    return checkpoint_prefix, checkpoint


# `tf.function` annotation causes the function to be "compiled".
@tf.function
def train_step(images, discriminator, generator, batch_size, 
                noise_dim, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Return a dictionary with metric names as keys and metric results as values

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig(f'./saved_images/image_at_epoch_{epoch:04d}.png')
    plt.show()

def train(generator, discriminator, train_ds, epochs, 
            batch_size, noise_dim, generator_optimizer, 
            discriminator_optimizer, checkpoint, 
            checkpoint_prefix, seed):

    for epoch in range(epochs):
        start = time.time()
        print(f'Epoch: {epoch + 1}/{epochs}')

        for image_batch in tqdm.tqdm(train_ds, position=0, leave=True):
            train_step(image_batch, discriminator, generator, batch_size,
                        noise_dim, generator_optimizer, discriminator_optimizer)

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            generate_and_save_images(generator, epoch, seed)
        
        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

    # Generate after the final epoch 
    # display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed)

# Display a single image using the epoch number
def display_image(epoch_number):
    return PIL.Image.open(f'./saved_images/image_at_epoch_{epoch_number:04d}.png')

def main():
    BATCH_SIZE = 128
    BUFFER_SIZE = 10000

    train_dataset = prepare_dataset(train_images, BATCH_SIZE, BUFFER_SIZE)

    generator = Generator()
    noise = tf.random.normal([1, 100])
    generated_image = generator(noise, training=False)
    plt.imshow(generated_image[0, :, :, 0])

    discriminator = Discriminator()
    decision = discriminator(generated_image)
    print(decision)

    # Define optimizers
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    epochs = 50
    noise_dim = 100
    num_examples_to_generate = 16
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    checkpoint_prefix, checkpoint = saves_checkpoints(generator, 
                                                        discriminator,
                                                        generator_optimizer, 
                                                        discriminator_optimizer)

    train(generator=generator,
            discriminator=discriminator,
            train_ds=train_dataset,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            noise_dim=noise_dim,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            checkpoint=checkpoint,
            checkpoint_prefix=checkpoint_prefix,
            seed=seed)

    display_image(epochs)
    
if __name__ == '__main__':
    main()