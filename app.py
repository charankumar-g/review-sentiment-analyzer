import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Clean review text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# 🎨 Inject Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rubik', sans-serif;
        background: linear-gradient(135deg, #f0f4f8, #d9e4f5);
        color: #222;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.65);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 40px;
        margin-top: 30px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #555;
        margin-bottom: 40px;
    }

    textarea {
        font-size: 16px !important;
        padding: 12px !important;
        border-radius: 10px !important;
    }

    .stButton > button {
        background: linear-gradient(to right, #4CAF50, #2ecc71);
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease-in-out;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #45a049, #27ae60);
        transform: scale(1.05);
    }

    .sentiment-box {
        font-size: 24px;
        font-weight: bold;
        padding: 20px;
        margin-top: 30px;
        border-radius: 12px;
        text-align: center;
        color: white;
        animation: fadeIn 1s ease-in-out;
    }

    .positive {
        background: linear-gradient(to right, #56ab2f, #a8e063);
    }

    .negative {
        background: linear-gradient(to right, #cb2d3e, #ef473a);
    }

    .neutral {
        background: linear-gradient(to right, #2193b0, #6dd5ed);
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)


# 📝 App Header
st.markdown('<div class="title">📝 Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze customer reviews with style and precision</div>', unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 🗣️ Input
review_input = st.text_area("💬 Enter a customer review:")

# 🔍 Predict
if st.button("Analyze Sentiment"):
    if review_input:
        cleaned_review = clean_text(review_input)
        review_vectorized = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vectorized)
        sentiment = prediction[0]

        # 🎨 Colorful Output Box
        emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}
        style_class = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral"
        }

        st.markdown(f"""
            <div class="sentiment-box {style_class[sentiment]}">
                {emoji[sentiment]} Sentiment: {sentiment.capitalize()}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a review before analyzing.")

st.markdown('</div>', unsafe_allow_html=True)
