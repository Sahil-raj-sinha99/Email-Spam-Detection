import streamlit as st
import re
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")

# helpers
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in stop_words)

# cache model loading to avoid reloading on every interaction
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("spam_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()
MAX_LEN = 120  # same as used for training

st.title("📧 Email Spam Classifier")
st.write("Enter an email message below:")

user_input = st.text_area("", height=200)
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        seq = tokenizer.texts_to_sequences([clean_text(user_input)])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
        pred = model.predict(pad)[0][0]
        label = "💀 Spam" if pred > 0.5 else "✅ Ham"
        st.markdown(f"### Prediction: **{label}**  \n(Probability: {pred:.3f})")