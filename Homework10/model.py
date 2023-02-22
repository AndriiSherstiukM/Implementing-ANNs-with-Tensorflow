import io
import re
import string
import tqdm
import numpy as np 
import tensorflow as tf

from tensorflow import keras
from keras import datasets, layers
from keras import models, losses

# Create Skip Gram model class
class SkipGram(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(SkipGram, self).__init__()
        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="w2v_embedding")
        self.context_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=num_ns)

    def call(self, pair):
        target, context = pair
        # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
        # context: (batch, context)
        if len(target.shape) == 2:
            target = tf.squeeze(target, axis=1)
        # target: (batch,)
        word_emb = self.target_embedding(target)
        # word_emb: (batch, embed)
        context_emb = self.context_embedding(context)
        # context_emb: (batch, context, embed)
        dots = tf.einsum('be,bce->bc', word_emb, context_emb)
        # dots: (batch, context)
        return dots

# Define loss function
def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)


