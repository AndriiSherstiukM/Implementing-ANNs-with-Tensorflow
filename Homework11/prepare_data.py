import re
import string
import tensorflow as tf
import matplotlib.pyplot as plt

from homework11 import vectorize_layer

# Create a custom standardization function to lowercase the text and
# remove special characters and punctuation.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    # lowercase = lowercase.split()
    return tf.strings.regex_replace(lowercase,
                                    '[%s]' % re.escape(string.punctuation), '')

def prepare_input_labels(data):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """

    data = tf.expand_dims(data, axis=-1)
    tokenized_sentences = vectorize_layer(data)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]

    return x, y

# Configure the dataset for performance
def config_dataset(text_ds, buffer_size, batch_size):
    text_ds = text_ds.shuffle(buffer_size=32)
    text_ds = text_ds.batch(batch_size, drop_remainder=True)
    text_ds = text_ds.map(prepare_input_labels)
    # Apply Dataset.cache and Dataset.prefetch to improve performance:
    text_ds = text_ds.cache()
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)

    return text_ds
