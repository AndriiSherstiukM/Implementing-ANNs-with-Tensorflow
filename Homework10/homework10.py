import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import datetime

from tensorflow import keras
from keras import datasets, layers
from keras import models, losses
from keras.utils import pad_sequences
from model import *
from prepare_data import * 

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
SEQUENCE_LENGTH = 4
# Set the number of negative samples per positive context.
NUM_NS = 4
WINDOW_SIZE = 2
SEED = 42
EMBEDDING_DIM = 64

def main():
    # Loading data 
    file_path = '../Homework10/bible.txt'

    with open(file_path, "r") as f:
        text = f.read().splitlines()

    for line in text[:20]:
        print(line) 

    # Preparing Data for Model
    text_ds = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    vocabulary = dict((x, i) for i, x in enumerate(np.unique(list(text))))
    vocab_size = len(vocabulary)
    print(vocab_size, '\n')

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)

    # Create the vocabulary for text dataset
    vectorize_layer.adapt(text_ds.batch(BATCH_SIZE))
    # Returns a list of all vocabulary tokens sorted (descending) by their frequency
    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:20], '\n')

    # Vectorize the data in text_ds.
    text_vector_ds = text_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE).map(vectorize_layer).unbatch()

    # Flatten the dataset into a list of sentence vector sequences
    sequences = list(text_vector_ds.as_numpy_iterator())
    print(len(sequences), '\n')
    # Inspect a few examples from sequences
    for seq in sequences[:7]:
        print(f'{seq} => {[inverse_vocab[i] for i in seq]}')
    print('\n')

    # Generate training examples from sequences
    targets, contexts, labels = generate_training_data(
        sequences=sequences,
        window_size=WINDOW_SIZE,
        num_ns=NUM_NS,
        vocab_size=vocab_size,
        seed=SEED)
    
    targets = np.array(targets)
    contexts = np.array(contexts)
    labels = np.array(labels)

    print('\n')
    print(f'targets shape: {targets.shape}')
    print(f'contexts shape: {contexts.shape}')
    print(f'labels shape: {labels.shape}')
    print('\n')

    dataset = config_dataset(targets, contexts, labels, BUFFER_SIZE, BATCH_SIZE)
    print(dataset)

    skip_gram = SkipGram(vocab_size, EMBEDDING_DIM, NUM_NS)
    skip_gram.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy'])

    # Define a callback to log training statistics for TensorBoard
    EXPERIMENT_NAME = 'Word embedding'
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(f'./logs/{EXPERIMENT_NAME}/{current_time}')

    skip_gram.fit(
        dataset, 
        epochs = 20,
        callbacks=[tensorboard_callback])
    
    # Let's take a look at a summary of Skip Gram model
    skip_gram.summary()

    # Obtain the weights from the model using Model.get_layer and Layer.get_weights.
    weights = skip_gram.get_layer('w2v_embedding').get_weights()[0]
    vocab = vectorize_layer.get_vocabulary()

    # Create and save the vectors and metadata files
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocab):
        if index == 0:
            continue  # skip 0, it's padding.
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()

if __name__ == '__main__':
    main()