import io
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras.layers import TextVectorization
from model import *
from prepare_data import *
from text_generator_callback import * 

BATCH_SIZE = 128
SEQUENCE_LENGTH = 64
# Set the number of negative samples per positive context.
NUM_HEADS = 2
SEED = 42
FEED_FORWARD_DIM = 64

buffer_size = 5000
embedding_dim = 64


def main():
    # Loading data
    file_path = '../Homework10/bible.txt'

    with open(file_path, "r") as f:
        text = f.read().splitlines()

    for line in text[:50]:
        print(line) 

    # Preparing Data for Model
    vocabulary = dict((x, i) for i, x in enumerate(np.unique(list(text))))
    vocab_size = len(vocabulary)
    print(vocab_size, '\n')

    # Preparing Data for Model
    dataset = tf.data.TextLineDataset(file_path).filter(lambda x: tf.cast(tf.strings.length(x), bool))

    # Use the `TextVectorization` layer to normalize, split, and map strings to
    # integers. Set the `output_sequence_length` length to pad all samples to the
    # same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH + 1)

    # Create the vocabulary for text dataset
    vectorize_layer.adapt(dataset.batch(BATCH_SIZE))
    # Returns a list of all vocabulary tokens sorted (descending) by their frequency
    inverse_vocab = vectorize_layer.get_vocabulary()
    print(inverse_vocab[:30], '\n')
    
    dataset = config_dataset(dataset, buffer_size, BATCH_SIZE)

    word_to_index = {}
    # Tokenize starting prompt
    for index, word in enumerate(inverse_vocab):
        word_to_index[word] = index

    start_prompt = 'The First Book'
    start_tokens = [word_to_index.get(_, 1) for _ in start_prompt.split()]
    num_tokens_generated = 50
    text_generator_callback = TextGenerator(num_tokens_generated, start_tokens, 
                                            inverse_vocab, SEQUENCE_LENGTH)
    
    gpt_model = createTransformerModel(vocab_size=vocab_size, 
                                    sequence_length=SEQUENCE_LENGTH, 
                                    embed_dim=embedding_dim,
                                    num_heads=NUM_HEADS,
                                    feed_forward_dim=FEED_FORWARD_DIM)
    
    history = gpt_model.fit(dataset,
                            epochs=25,
                            verbose=1,
                            callbacks=[text_generator_callback])
    
    gpt_model.summary()

    # Obtain the weights from the model using Model.get_layer and Layer.get_weights.
    weights = gpt_model.layers[1].get_weights()
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