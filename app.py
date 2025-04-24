import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load the pre-trained model and vectorizer (adjust path if needed)
model = joblib.load('sentiment_model.pkl')  # Load your trained model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load your fitted TF-IDF vectorizer


# Function to clean the text (same as your previous code)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>', '', text)  # remove HTML
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation and numbers
    return text


# Streamlit app title and description
st.title('Sentiment Analysis for Customer Reviews')
st.write('Enter a review and see if it is positive, neutral, or negative.')

# Text input for user
review_input = st.text_area("Enter a review:")

# Button to predict sentiment
if st.button('Analyze Sentiment'):
    if review_input:
        cleaned_review = clean_text(review_input)
        review_vectorized = vectorizer.transform([cleaned_review])  # Transform the cleaned review
        prediction = model.predict(review_vectorized)  # Predict sentiment

        sentiment = prediction[0]
        st.write(f"Sentiment: {sentiment.capitalize()}")
    else:
        st.write("Please enter a review to analyze.")

import joblib

# Save your trained model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
