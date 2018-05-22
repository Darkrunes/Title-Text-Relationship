import keras
import tensorflow as tf
import numpy as np
import fake_news

MAX_SEQ_LEN = 200
EMBEDDING_DIMENSIONS = 50
VALIDATION_SPLIT = 0.2

model = keras.models.load_model("Checkpoints/weights.05.hdf5")

header = input("Enter Header: ")
body = input("Enter body: ")

article = header + " " + body
article = fake_news.preprocess_text([article])

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(article)
seqs = tokenizer.texts_to_sequences(article)

article = keras.preprocessing.sequence.pad_sequences(seqs, MAX_SEQ_LEN)

result = model.predict(article)

print(result)
res = np.argmax(result)

print(res)
if res == 1:
    print("The body agrees with the header")
elif res == 2:
    print("The body disagrees with the header")
elif res == 3:
    print("The body discusses with the header")
elif res == 4:
    print("The body is unrelated to the header")

sess = tf.keras.backend.get_session()
sess.close()

