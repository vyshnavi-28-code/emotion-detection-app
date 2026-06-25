import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("training.csv")

# Features and labels
X = df['text']
y = df['label']

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Emotion mapping
emotion_map = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}

# UI
st.title("🧠 Emotion Detection App")
st.write("Enter a sentence and find its emotion!")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)[0]
        st.success(f"Predicted Emotion: {emotion_map[prediction]}")
    else:
        st.warning("Please enter some text")
