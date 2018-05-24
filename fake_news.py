# Fake News Challenge Attempt using RNN
# Saffat Shams Akanda, z5061498 @ UNSW

import keras
import pandas as pd
import numpy as np
import os
import re
import rnn_model

MAX_SEQ_LEN = 200
EMBEDDING_DIMENSIONS = 50
VALIDATION_SPLIT = 0.2

TRAINING_DIR = "training_data/"
EMBEDDING_NAME = "glove.6B.50d.txt"


def read_embeddings():
    if os.path.exists(EMBEDDING_NAME):
        file = open(EMBEDDING_NAME, "r", encoding="utf-8")
    else:
        raise Exception("No Glove File found in directory, requires " + EMBEDDING_NAME)

    embeddings_index = dict()
    word_index_dict = dict()
    count = 1
    for line in file:
        currline = line.split()
        word = currline[0]
        embeddings_index[word] = np.asarray(currline[1:], dtype='float32')
        word_index_dict[word] = count
        count += 1

    file.close()
    return embeddings_index, word_index_dict


def read_input_files():
    labels = []
    text = []
    with open(TRAINING_DIR + "train_stances.csv", "r", encoding="utf-8") as file:
        train_bodies = pd.read_csv(TRAINING_DIR + "train_bodies.csv", index_col="Body ID")

        next(file)                          # First line has header info
        for line in file:
            values = line.rsplit(",", 2)
            values[2] = values[2].strip()
            if values[2] == "agree":
                labels.append(1)
            elif values[2] == "disagree":
                labels.append(2)
            elif values[2] == "discuss":
                labels.append(3)
            elif values[2] == "unrelated":
                labels.append(4)

            text.append(values[0].strip() + " " + (train_bodies.at[int(values[1]), "articleBody"]).strip())

    return preprocess_text(text), labels


def preprocess_text(text_arr):
    maxlen = 0
    k = 0
    for i in range(len(text_arr)):
        if len(text_arr[i]) > maxlen:
            maxlen = len(text_arr[i])
            k = i
        text_arr[i] = re.sub(r'[^\w ]', '', text_arr[i])

    return text_arr


def main():
    # Read in Embeddings
    embeddings_index, word_index_dict = read_embeddings()
    # Read in input files
    train_text, train_labels = read_input_files()

    # Tokenize text
    print("Tokenizing")
    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(train_text)
    seqs = tokenizer.texts_to_sequences(train_text)
    word_index = tokenizer.word_index

    padded_seq = keras.preprocessing.sequence.pad_sequences(seqs, MAX_SEQ_LEN)
    train_labels = keras.utils.to_categorical(np.asarray((train_labels)))
    print("Data tensor: ", padded_seq.shape)
    print("Label Tensor: ", train_labels.shape)

    # Split into test and validation sets
    print("Creating Sets")
    indicies = np.arange(padded_seq.shape[0])
    np.random.shuffle(indicies)
    data = padded_seq[indicies]
    train_labels = train_labels[indicies]
    validation_set_size = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-validation_set_size]
    y_train = train_labels[:-validation_set_size]
    x_val = data[-validation_set_size:]
    y_val = train_labels[-validation_set_size:]

    # Create embeddings matrix to use as input
    print("Turning into embeddings")
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIMENSIONS))
    for word, i in word_index.items():
        #if i >= :
        #    continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is None:
            embedding_matrix[i] = embeddings_index.get("unk")
        else:
            embedding_matrix[i] = embedding_vector

    print("Making the model")
    nn_model = rnn_model.recurrentModel()
    nn_model.combined_model(num_words, embedding_matrix, MAX_SEQ_LEN, EMBEDDING_DIMENSIONS)
    nn_model.train(x_train, y_train, x_val, y_val)

if __name__ == "__main__":
    main()
