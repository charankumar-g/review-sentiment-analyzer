import streamlit as st
import re
import joblib

# Load pre-trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --- Clean text function ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# --- CSS for styling ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500&display=swap');

        body {
            background-color: #f4f6f8;
        }
        .main-title {
            font-size: 38px;
            font-weight: bold;
            background: -webkit-linear-gradient(45deg, #6a11cb, #2575fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 5px;
        }
        .sub-title {
            font-size: 20px;
            color: #666;
            margin-bottom: 30px;
        }
        textarea {
            font-size: 16px !important;
            padding: 12px !important;
            line-height: 1.6 !important;
            border-radius: 12px !important;
            background-color: #ffffff;
        }
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            font-size: 16px;
            padding: 12px 28px;
            border: none;
            border-radius: 12px;
            transition: 0.3s ease;
            font-family: 'Poppins', sans-serif;
        }
        div.stButton > button:first-child:hover {
            background: linear-gradient(90deg, #2575fc, #6a11cb);
            transform: scale(1.03);
            color: white;
        }
        .positive {
            background: linear-gradient(90deg, #00c851, #007e33);
        }
        .negative {
            background: linear-gradient(90deg, #ff4444, #cc0000);
        }
        .neutral {
            background: linear-gradient(90deg, #33b5e5, #0099cc);
        }
        .sentiment-inline {
            font-size: 16px;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 10px;
            color: white;
            display: inline-block;
            margin-left: 15px;
            vertical-align: middle;
        }
        .footer {
            font-size: 12px;
            color: #aaa;
            margin-top: 60px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-title">💬 Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Instantly analyze customer reviews with AI-powered sentiment prediction</div>', unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #e0e0e0;'>", unsafe_allow_html=True)

# --- Input ---
review_input = st.text_area("Write your review here:")

# --- Button and Result Side-by-Side Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    analyze_clicked = st.button("Analyze Sentiment")

with col2:
    if analyze_clicked and review_input.strip():
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Style based on prediction
        sentiment_class = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral'
        }.get(prediction.lower(), 'neutral')

        # Display inline beside button
        st.markdown(f"<div class='sentiment-inline {sentiment_class}'>Sentiment: {prediction.capitalize()}</div>", unsafe_allow_html=True)

if analyze_clicked and not review_input.strip():
    st.warning("⚠️ Please enter a review to analyze.")

# --- Footer ---
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
