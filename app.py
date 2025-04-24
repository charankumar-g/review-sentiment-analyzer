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
        .sentiment-box {
            font-size: 22px;
            font-weight: bold;
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            margin-top: 25px;
            display: inline-block;
            animation: fadeIn 0.6s ease-in-out;
        }
        @keyframes fadeIn {
            0% {opacity: 0; transform: translateY(10px);}
            100% {opacity: 1; transform: translateY(0);}
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


# --- Prediction + Dynamic Result Style ---
if st.button("Analyze Sentiment"):
    if review_input.strip():
        cleaned = clean_text(review_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Determine sentiment class for styling
        css_class = {
            'positive': 'positive',
            'negative': 'negative',
            'neutral': 'neutral'
        }.get(prediction.lower(), 'neutral')

        # Display sentiment with gradient color
        st.markdown(
            f"<div class='sentiment-box {css_class}'>Sentiment: {prediction.capitalize()}</div>",
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ Please enter a review to analyze.")


# --- Footer ---
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)
