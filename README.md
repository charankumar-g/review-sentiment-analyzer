Great! Here's an updated version of your `README.md`, incorporating the data source from Kaggle:

---

## 💬 Review Sentiment Analyzer

An interactive web app that uses Natural Language Processing and Machine Learning to classify customer reviews as **Positive**, **Negative**, or **Neutral** in real-time.

🔥 [Check out the live app!](https://review-sentiment-analyzer-9xq56pwgjvlw6kxspbzzfd.streamlit.app/)

---

## ✨ Features

- 🧠 Built with Logistic Regression + TF-IDF Vectorizer
- 🧹 Custom text preprocessing and cleaning
- 🎨 Stylish, responsive UI with gradients and animations
- 📦 Deployed using Streamlit Cloud
- ⚡ Instant predictions for any review text

---

## 📚 Dataset

- **Source**: [Amazon Food Reviews Dataset from Kaggle](https://www.kaggle.com/datasets)
- **Data Fields**:
  - Review text
  - Rating
  - Product details
- **Key Insights**:
  - The dataset provides a wide range of Amazon food reviews to predict sentiment based on the review text.

---

## 🖼️ Preview

![App Screenshot](
app_screenshot.PNG)

---

## 🛠️ How to Run Locally

1. Clone this repo
   ```bash
   git clone https://github.com/charankumar-g/review-sentiment-analyzer.git
   cd review-sentiment-analyzer
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app
   ```bash
   streamlit run app.py
   ```

---

## 📂 Project Structure

```
├── app.py                  # Streamlit app code
├── sentiment_model.pkl     # Trained logistic regression model
├── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer
├── cleaned_reviews_sample.csv  # Sample cleaned review data
├── .streamlit/config.toml  # Custom Streamlit theme
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🚀 Tech Stack

- Python
- Pandas, Scikit-learn, Regex
- Streamlit
- Logistic Regression
- TF-IDF (Text Vectorization)

---

---

## 🔗 Connect with Me

- [GitHub](https://github.com/charankumar-g)
- [LinkedIn] (https://www.linkedin.com/in/charan-kumar-g/)

---
