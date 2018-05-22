# Fake News Challenge Attempt using RNN
# Saffat Shams Akanda, z5061498 @ UNSW

import keras
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM
import numpy
import os

RECURRENT_UNITS = 128
DROPOUT_RATE = 0.3
EPOCHS = 50


class recurrentModel():

    def __init__(self):
        pass

    def build_model_basic_RNN(self, num_words, embedding_matrix, max_seq_len, embedding_dimensions):
        self.modelInput = Input(shape=(max_seq_len, ), dtype="int32")
        embedding_layer = keras.layers.Embedding(num_words, embedding_dimensions,
                                                 weights=[embedding_matrix],
                                                 input_length=max_seq_len,
                                                 trainable=False)(self.modelInput)
        """
        x = Dense(512, activation="relu")(embedding_layer)

        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(256)(x)

        x = Dense(512, activation="relu")(x)
        x = Dense(512, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        self.preds = Dense(64, activation="softmax")(x)
        self.model = keras.Model(self.modelInput, self.preds)

        """
        x = Conv1D(128, 5, activation='relu')(embedding_layer)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        self.preds = Dense(4, activation='softmax')(x)

        self.model = keras.Model(self.modelInput, self.preds)
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def predict(self):
        pass

    def train(self, x_train, y_train, x_val, y_val):
        tbCallBack = keras.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
        checkpointCallBack = keras.callbacks.ModelCheckpoint("Checkpoints/weights.{epoch:02d}.hdf5", period=5)

        self.model.fit(x_train, y_train, batch_size=50, epochs=EPOCHS, validation_data=(x_val, y_val),
                       callbacks=[tbCallBack, checkpointCallBack])

    def build_model_basic_RNN2(self, num_words, embedding_matrix, max_seq_len, embedding_dimensions):

        self.modelInput = Input(shape=(max_seq_len, ), dtype="int32")
        embedding_layer = (keras.layers.Embedding(num_words, embedding_dimensions,
                                                  weights=[embedding_matrix],
                                                  input_length=max_seq_len,
                                                  trainable=False,
                                                  ))(self.modelInput)

        x = LSTM(RECURRENT_UNITS, return_sequences=True)(embedding_layer)
        x = LSTM(RECURRENT_UNITS, return_sequences=True, dropout=DROPOUT_RATE)(x)
        x = LSTM(RECURRENT_UNITS, dropout=DROPOUT_RATE)(x)

        x = Dense(512, activation="relu")(x)
        x = keras.layers.Dropout(DROPOUT_RATE)(x)
        x = Dense(256, activation="relu")(x)
        x = keras.layers.Dropout(DROPOUT_RATE)(x)
        x = Dense(128, activation="relu")(x)

        self.preds = Dense(5, activation="softmax")(x)
        self.model = keras.Model(self.modelInput, self.preds)

        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def build_model_basic_RNN3(self, num_words, embedding_matrix, max_seq_len, embedding_dimensions):


        self.model = keras.Sequential()
        self.model.add(keras.layers.Embedding(num_words, embedding_dimensions,
                                              weights=[embedding_matrix],
                                              input_length=max_seq_len,
                                              trainable=False,
                                              ))

        self.model.add(LSTM(256))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(5, activation="softmax"))

        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                           metrics=["accuracy"])
