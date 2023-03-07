import numpy as np
import tensorflow as tf

from tensorflow import keras

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """
    def __init__(self, max_tokens, start_tokens, index_to_word, sequence_length, k=10, print_every=1):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.k = k
        self.print_every = print_every
        self.sequence_length = sequence_length

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, self.k, sorted=True)
        indices = np.asarray(indices).astype('int32')
        predictions = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        predictions = np.asarray(predictions).astype('float32') 

        return np.random.choice(indices, p=predictions)
    
    def detokenize(self, number):
        return self.index_to_word[number]
    
    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.sequence_length - len(start_tokens)
            sample_index = len(start_tokens) - 1

            if pad_len < 0:
                x = start_tokens[:self.sequence_length]
                sample_index = self.sequence_length - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens

            x = np.array([x])
            y = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        
        text = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated]) 
        print(f'Generated text: {text}\n')

