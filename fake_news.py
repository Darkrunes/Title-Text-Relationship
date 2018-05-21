# Fake News Challenge Attempt using RNN
# Saffat Shams Akanda, z5061498 @ UNSW

import keras
import tensorboard
import pandas as pd
import numpy as np
import os
import rnn_model

MAX_SEQ_LEN = 250
EMBEDDING_DIMENSIONS = 200
VALIDATION_SPLIT = 0.2

TRAINING_DIR = "training_data/"
EMBEDDING_NAME = "glove.6B.200d.txt"


def read_embeddings():
    if os.path.exists(EMBEDDING_NAME):
        file = open(EMBEDDING_NAME, "r", encoding="utf-8")
    else:
        raise Exception("No Glove File found in directory, requires " + EMBEDDING_NAME)

    embeddings_index = dict()
    for line in file:
        currline = line.split()
        word = currline[0]
        embeddings_index[word] = np.asarray(currline[1:], dtype='float32')

    file.close()
    return embeddings_index


def read_input_files():
    labels = []
    text = []
    with open(TRAINING_DIR + "train_stances.csv", "r", encoding="utf-8") as file:
        train_bodies = pd.read_csv(TRAINING_DIR + "train_bodies.csv", index_col="Body ID")

        next(file)                          # First line has header info
        for line in file:
            values = line.rsplit(",", 2)
            labels.append(values[2])
            text.append(values[0] + " " + train_bodies.at[int(values[1]), "articleBody"])

    return text, labels


def main():
    print("Reading in embeddings")
    embeddings_index = read_embeddings()
    print("Reading in input files now")
    train_text, train_labels = read_input_files()

    print("I CAN READ")


if __name__ == "__main__":
    main()
