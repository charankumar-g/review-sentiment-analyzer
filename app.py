import streamlit as st
import re
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')


# --- Text cleaning function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# --- CSS Styling for modern UI ---
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        .main-title {
            font-size: 38px;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #ff4b2b, #ff416c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 20px;
            color: #555;
            margin-bottom: 30px;
        }
        textarea {
            font-size: 16px !important;
            padding: 12px !important;
            line-height: 1.6 !important;
            border-radius: 10px !important;
        }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #ff416c, #ff4b2b);
            color: white;
            font-size: 16px;
            padding: 12px 28px;
            border: none;
            border-radius: 12px;
            transition: 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #ff4b2b, #ff416c);
            color: white;
            transform: scale(1.03);
        }
        .result-text {
            font-size: 22px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 8px;
            background-color: #ffffff;
            border-left: 6px solid #ff416c;
            margin-top: 20px;
            color: #333;
        }
    </style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown('<div class="main-title">💬 Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instantly analyze customer reviews with AI-powered sentiment prediction</div>', unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #e0e0e0;'>", unsafe_allow_html=True)


# --- Text Input ---
review_input = st.text_area("Write your review here:")


# --- Prediction ---
if st.button("Analyze Sentiment"):
    if review_input.strip():
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        st.markdown(
            f"<div class='result-text'>Sentiment: <span style='color:#ff4b2b'>{prediction.capitalize()}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a review to analyze.")
