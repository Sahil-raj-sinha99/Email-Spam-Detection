import streamlit as st
import re
import pickle
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    stop_words = set(stopwords.words("english"))
    return " ".join(w for w in text.split() if w not in stop_words)

@st.cache_resource
def load_artifacts():
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("📧 Email Spam Classifier")
st.write("Enter an email message below:")

user_input = st.text_area("", height=200)
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please type something first.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]
        label = "💀 Spam" if pred == 1 else "✅ Ham"
        st.markdown(f"### Prediction: **{label}**  \n(Spam Probability: {prob:.3f})")