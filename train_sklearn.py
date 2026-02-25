import numpy as np
import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Common English stopwords (no NLTK needed)
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and',
    'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below',
    'between', 'both', 'but', 'by', 'can', 'cannot', 'could', 'did', 'do', 'does',
    'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'has',
    'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'might',
    'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only',
    'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'so', 'some',
    'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
    'these', 'they', 'this', 'those', 'to', 'too', 'under', 'until', 'up', 'very', 'was',
    'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will',
    'with', 'you', 'your', 'yours', 'yourself', 'yourselves'
}

# LOAD DATA
data = pd.read_csv('Emails.csv')
print(data.head())

# CLEAN TEXT FUNCTION
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join(w for w in text.split() if w not in STOPWORDS)

data["Message"] = data["Message"].apply(clean_text)

# BALANCE DATASET
ham = data[data["Category"] == "ham"]
spam = data[data["Category"] == "spam"]
ham_bal = ham.sample(len(spam), random_state=42)
balanced = pd.concat([ham_bal, spam]).reset_index(drop=True)

# PREPARE DATA
X = balanced["Message"].values
y = (balanced["Category"] == "spam").astype(int)

train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

# VECTORIZE TEXT
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
train_vec = vectorizer.fit_transform(train_X)
test_vec = vectorizer.transform(test_X)

# TRAIN MODEL
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(train_vec, train_Y)

# EVALUATE
pred = model.predict(test_vec)
acc = accuracy_score(test_Y, pred)
print(f"Test Accuracy: {acc:.4f}")

# SAVE MODEL AND VECTORIZER
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)
    
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved as 'spam_model.pkl'")
print("✅ Vectorizer saved as 'vectorizer.pkl'")
