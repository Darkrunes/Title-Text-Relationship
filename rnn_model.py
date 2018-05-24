# Fake News Challenge Attempt using RNN
# Saffat Shams Akanda, z5061498 @ UNSW

import keras
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, LSTM, Dropout
import numpy
import os

RECURRENT_UNITS = 128
DROPOUT_RATE = 0.4
EPOCHS = 50


class recurrentModel():

    def __init__(self):
        pass

    def combined_model(self, num_words, embedding_matrix, max_seq_len, embedding_dimensions):
        self.modelInput = Input(shape=(max_seq_len, ), dtype="int32")
        embedding_layer = keras.layers.Embedding(num_words, embedding_dimensions,
                                                 weights=[embedding_matrix],
                                                 input_length=max_seq_len,
                                                 trainable=False)(self.modelInput)
        y = Conv1D(128, 5, activation='relu')(embedding_layer)
        y = MaxPooling1D(4)(y)
        y = Conv1D(64, 5, activation='relu')(y)
        y = Conv1D(128, 5, activation='relu')(y)
        y = MaxPooling1D(4)(y)

        x = LSTM(RECURRENT_UNITS, return_sequences=True, dropout=DROPOUT_RATE)(y)
        x = LSTM(RECURRENT_UNITS, return_sequences=True, dropout=DROPOUT_RATE)(x)
        x = LSTM(RECURRENT_UNITS, dropout=DROPOUT_RATE)(x)

        z = Dense(512)(x)
        z = Dropout(DROPOUT_RATE)(z)
        z = Dense(512)(z)
        self.preds = Dense(5, activation="softmax")(z)
        self.model = keras.Model(self.modelInput, self.preds)

        self.model.compile(optimizer="adam", loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def build_model_basic_CNN(self, num_words, embedding_matrix, max_seq_len, embedding_dimensions):
        self.modelInput = Input(shape=(max_seq_len, ), dtype="int32")
        embedding_layer = keras.layers.Embedding(num_words, embedding_dimensions,
                                                 weights=[embedding_matrix],
                                                 input_length=max_seq_len,
                                                 trainable=False)(self.modelInput)

        x = Conv1D(128, 5, activation='relu')(embedding_layer)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128, 5, activation='relu')(x)
        x = keras.layers.GlobalMaxPooling1D()(x)
        x = Dense(128, activation='relu')(x)
        self.preds = Dense(4, activation='softmax')(x)

        self.model = keras.Model(self.modelInput, self.preds)
        self.model.compile(optimizer="adam", loss="categorical_crossentropy",
                           metrics=["accuracy"])

    def predict(self, text):
        return self.model.predict(text)

    def train(self, x_train, y_train, x_val, y_val):
        tbCallBack = keras.callbacks.TensorBoard(log_dir="./logs", write_graph=True, write_images=True)
        checkpointCallBack = keras.callbacks.ModelCheckpoint("./Checkpoints/model_and_weights.{epoch:02d}.hdf5",
                                                             period=5)
        loss_history = LossHistory()

        self.model.fit(x_train, y_train, batch_size=50, epochs=EPOCHS, validation_data=(x_val, y_val),
                       callbacks=[tbCallBack, checkpointCallBack, loss_history])

        with open("./Checkpoints/Epoch_Information.txt", "w") as f:
            for i in range(len(loss_history.losses)):
                f.write("Epoch " + str(i))
                f.write("Loss: " + str(loss_history.losses[i]))
                f.write("Validation Loss: " + str(loss_history.val_losses[i]))

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


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


