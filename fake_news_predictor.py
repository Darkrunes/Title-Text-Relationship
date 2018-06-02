import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import re

MAX_SEQ_LEN = 200


def read_input_files():
    text = []
    print("Reading input files")
    with open(sys.argv[1], "r", encoding="utf-8") as file:
        train_bodies = pd.read_csv(sys.argv[2], index_col="Body ID")

        i = 0
        for line in file:
            if i == 0:
                header_line = line
                i+=1
                continue

            values = line.rsplit(",", 1)

            i += 1
            text.append(values[0].strip() + " " + (train_bodies.at[int(values[1]), "articleBody"]).strip())

    return preprocess_text(text), header_line


def preprocess_text(text_arr):
    maxlen = 0
    for i in range(len(text_arr)):
        if len(text_arr[i]) > maxlen:
            maxlen = len(text_arr[i])
        text_arr[i] = re.sub(r'[^\w ]', '', text_arr[i])
        text_arr[i] = re.sub(r'(\n)', '', text_arr[i])

    return text_arr

test_text, header_line = read_input_files()
model = keras.models.load_model("combined_model.hdf5")

print("Tokenizing")
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(test_text)
seqs = tokenizer.texts_to_sequences(test_text)

articles = keras.preprocessing.sequence.pad_sequences(seqs, MAX_SEQ_LEN)

answers = []
print("Making predictions")
for article in articles:
    art = np.array([article])
    result = model.predict(art)
    res = np.argmax(result)
    if res == 1:
        answers.append(["agree"])
    elif res == 2:
        answers.append(["disagree"])
    elif res == 3:
        answers.append(["discuss"])
    elif res == 4:
        answers.append(["unrelated"])

print("Writing to CSV")
lines = []
with open(sys.argv[1], "r", encoding="utf-8") as file:
    for line in file:
        lines.append(line.strip())

with open("predicted_stances.csv", "w", encoding="utf-8") as file:
    file.write(lines[0] + ",Stances\n")
    for i in range(1, len(lines)):
        file.write(str(lines[i]) + "," + str(answers[i-1][0]) + "\n")


sess = tf.keras.backend.get_session()
sess.close()

