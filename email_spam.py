import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('stopwords')

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

# LOAD DATA
data = pd.read_csv('Emails.csv')
print(data.head())

# CLEAN TEXT FUNCTION
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)         # Remove URLs
    text = re.sub(r"\d+", "", text)            # Remove numbers
    text = re.sub(r"[^\w\s]", "", text)        # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()   # Remove extra spaces
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data["Message"] = data["Message"].apply(clean_text)

# BALANCING DATASET
ham = data[data["Category"] == "ham"]
spam = data[data["Category"] == "spam"]

ham_bal = ham.sample(len(spam), random_state=42)
balanced = pd.concat([ham_bal, spam]).reset_index(drop=True)

sns.countplot(x='Category', data=balanced)
plt.title("Balanced Data")
plt.show()

# TOKENIZATION
X = balanced["Message"].values
y = (balanced["Category"] == "spam").astype(int)

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(train_X)

train_seq = tokenizer.texts_to_sequences(train_X)
test_seq = tokenizer.texts_to_sequences(test_X)

max_len = 120
train_pad = pad_sequences(train_seq, maxlen=max_len, padding='post')
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')

# IMPROVED MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=64, input_length=max_len),
    tf.keras.layers.SpatialDropout1D(0.3),
    
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# CALLBACKS
es = EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# TRAIN MODEL
history = model.fit(
    train_pad, train_Y,
    validation_data=(test_pad, test_Y),
    epochs=10,
    batch_size=32,
    callbacks=[es, lr]
)

# EVALUATE
loss, acc = model.evaluate(test_pad, test_Y)
print("Test Accuracy:", acc)

# PLOT ACCURACY
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['Train','Validation'])
plt.show()