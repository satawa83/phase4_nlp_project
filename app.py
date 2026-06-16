import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# Set page title
st.set_page_config(page_title="Sentiment Analysis App", layout="centered")

st.title("🐦 Tweet Sentiment Analyzer")
st.write("Enter a tweet below and I'll predict if it's positive, negative, or neutral!")

# Load the saved models (these are in your folder)
@st.cache_resource
def load_models():
    try:
        model = joblib.load("sentiment_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Make sure these files exist in the same folder: sentiment_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")
        return None, None, None

model, vectorizer, label_encoder = load_models()

# Text input from user
user_input = st.text_area("✏️ Type your tweet here:", height=100)

# Predict button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please type a tweet first!")
    elif model is not None:
        # Preprocess and predict
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)
        sentiment = label_encoder.inverse_transform(prediction)[0]
        
        # Show result
        st.subheader("📊 Prediction Result:")
        
        # Color-coded output
        if sentiment.lower() == "positive":
            st.success(f"✅ {sentiment.upper()} 😊")
        elif sentiment.lower() == "negative":
            st.error(f"❌ {sentiment.upper()} 😠")
        else:
            st.info(f"⚪ {sentiment.upper()} 😐")
        
        # Show probability if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_vector)[0]
            st.write("Confidence Scores:")
            for i, cls in enumerate(label_encoder.classes_):
                st.write(f"- {cls}: {probs[i]:.2%}")
    else:
        st.error("Models not loaded properly. Please check your pickle files.")

# Show a helpful note
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")