import tensorflow as tf

from tensorflow import keras
from keras import layers, models
from homework11 import *

def casual_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Mask the upper half of the dot product matrix in self attention.
    This prevents flow of information from future tokens to current token.
    1's in the lower triangle, counting from the lower right corner.
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest

    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, axis=-1), tf.constant([1, 1], dtype=tf.int32)], axis=0
    )

    return tf.tile(mask, mult)

# Create a Transformer block as a layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Transformer block multi-head Self Attention
        self.multiHeadSelfAtt = layers.MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),
        ])
        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout_1 = layers.Dropout(dropout)
        self.dropout_2 = layers.Dropout(dropout)
        
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        casual_mask = casual_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        attention_output = self.multiHeadSelfAtt(inputs, inputs, attention_mask=casual_mask)
        attention_output = self.dropout_1(attention_output)
        out_1 = self.layer_norm_1(inputs + attention_output)
        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output)
        
        return self.layer_norm_2(out_1 + ffn_output)

# Create two separate embedding layers: one for tokens and one for token index (positions).
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(0, maxlen)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        return x + positions

def createTransformerModel(vocab_size, sequence_length, embed_dim, num_heads, feed_forward_dim):
    model = models.Sequential()
    model.add(layers.Input(shape=(sequence_length,), dtype=tf.int32))
    # Add a class with two separate embedding layers
    model.add(TokenAndPositionEmbedding(sequence_length, num_heads, feed_forward_dim))
    model.add(TransformerBlock(embed_dim, num_heads, feed_forward_dim))
    model.add(layers.Dense(units=vocab_size))

    loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=[loss_function, None]
    )
    # No loss and optimization based on word embeddings from transformer block
    return model