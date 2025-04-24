import streamlit as st
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# --- Clean Text Function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# --- Custom CSS Styling ---
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 20px;
            color: #555;
            margin-bottom: 30px;
        }
        textarea {
            font-size: 16px !important;
            padding: 10px !important;
            line-height: 1.5 !important;
        }
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #45a049;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)


# --- Header Titles ---
st.markdown('<div class="main-title">📝 Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze customer reviews with smart sentiment classification</div>', unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #e0e0e0;'>", unsafe_allow_html=True)


# --- Text Input ---
review_input = st.text_area("Enter a review:")


# --- Button and Prediction ---
submit = st.button("Analyze Sentiment")

if submit:
    if review_input:
        cleaned_review = clean_text(review_input)
        review_vectorized = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vectorized)
        sentiment = prediction[0]

        st.markdown(
            f"<div style='font-size:20px; color:#333; font-weight:bold;'>Sentiment: <span style='color:#4CAF50;'>{sentiment.capitalize()}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a review to analyze.")
