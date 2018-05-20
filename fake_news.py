# Fake News Challenge Attempt using RNN
# Saffat Shams Akanda, z5061498 @ UNSW

import tensorflow
import tensorboard
import numpy as np
import os
import rnn_model

MAX_SEQ_LEN = 2000
EMBEDDING_DIMENSIONS = 300
VALIDATION_SPLIT = 0.2


def read_embeddings():
    if os.path.exists("glove.840B.300d"):
        file = open("glove.840B.300d", "r", encoding="utf-8")
    else:
        raise Exception("No Glove File found in directory, requires glove.840B.300d")

    embeddings_index = dict()
    embeddings_index["UNK"] = 0
    for line in file:
        currline = line.split()
        word = currline[0]
        embeddings_index[word] = np.asarray(currline[1:], dtype='float32')

    file.close()
    return embeddings_index


def read_input_file():
    pass


def preprocess_file():
    pass


def main():
    pass


if __name__ == "__main__":
    main()
