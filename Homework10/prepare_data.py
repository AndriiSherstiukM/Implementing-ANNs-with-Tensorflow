import io
import re
import string
import tqdm
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime

from tensorflow import keras
from keras import datasets, layers
from keras import models, losses
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from keras.preprocessing.sequence import make_sampling_table
from keras.utils import pad_sequences

# Vectorize sentences from the corpus
# Now, create a custom standardization function to lowercase the text and
# remove special characters and punctuation.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    # lowercase = lowercase.split()
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation), '')

# Configure the dataset for performance
def config_dataset(targets, contexts, labels, 
                    buffer_size, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
    # Apply Dataset.cache and Dataset.prefetch to improve performance:
    dataset = dataset.cache().prefetch(buffer_size = tf.data.AUTOTUNE)

    return dataset

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, 
                            vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []
    # Build the sampling table for 'vocab_size' tokens.
    sampling_table = make_sampling_table(vocab_size)

    # Iterate over all sequences (sentences) in the dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size, 
            negative_samples=0
        )        

        # Iterate over each positive skip-gram pair to produce training examples
        # with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype='int64'), 1)
            negative_sample_candidate, _, _ = tf.random.log_uniform_candidate_sampler(
                true_classes=context_class,
                num_true=1,
                num_sampled=num_ns,
                unique=True,
                range_max=vocab_size,
                seed=seed,
                name='negative_sampling'
            )
        
            # Build context and label vectors (for one target word)
            context = tf.concat([tf.squeeze(context_class, 1), negative_sample_candidate], 0)
            label = tf.constant([1] + [0] * num_ns, dtype='int64')

            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)

    return targets, contexts, labels