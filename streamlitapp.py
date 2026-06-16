import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import html
import unicodedata
import contractions
import pickle
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# Define stopwords and helper functions from the notebook
common_english = {
    'the','be','to','of','and','a','in','that','have','i',
    'it','for','not','on','with','he','as','you','do','at',
    'this','but','his','by','from','they','we','say','her','she',
    'or','an','will','my','one','all','would','there','their',
    'what','so','up','out','if','about','who','get','which','go',
    'me','when','make','can','like','time','just','him','know',
    'take','people','into','year','your','good','some','could','them',
    'see','other','than','then','now','look','only','come','its','over',
    'think','also','back','after','use','two','how','our','work',
    'first','well','way','even','new','want','because','any','these',
    'give','day','most','us','is','was','are','been','has','had',
    'were','may','might','must','shall','should','will','would'
}

tech_words = {
    'iphone','ipad','ipod','mac','imac','macbook','apple','ios',
    'android','google','pixel','nexus','samsung','xiaomi','huawei',
    'oneplus','lg','sony','nokia','blackberry','windows','microsoft',
    'surface','xbox','playstation','nintendo','amazon','alexa','echo',
    'facebook','instagram','twitter','tiktok','snapchat','whatsapp',
    'telegram','signal','discord','reddit','linkedin','youtube',
    'netflix','spotify','uber','lyft','airbnb','paypal','venmo',
    'sxsw','sxswi','retweet','tweet','hashtag','viral','post','share','like'
}

stop_words = set(stopwords.words('english')) | common_english
lemmatizer = WordNetLemmatizer()

def is_meaningful_word(word):
    word = word.lower()
    if len(word) < 3:
        return False
    if not any(c.isalpha() for c in word):
        return False
    if not any(c in "aeiou" for c in word):
        return False
    if re.search(r'[^aeiou]{4,}', word):
        return False
    if re.search(r'(.)\\1{2,}', word):
        return False
    if word in tech_words:
        return True
    return word.isalpha()

def clean_text_comprehensive(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'^RT\\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'@\\w+', '', text)
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = html.unescape(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    text = re.sub(r'#(\\w+)', r'\\1', text)
    text = re.sub(r'\\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\\s+', ' ', text).strip()
    words = text.split()
    cleaned_words = []
    for w in words:
        if w in stop_words and w not in {"no", "not", "nor", "n't"}:
            continue
        if not is_meaningful_word(w):
            continue
        w = lemmatizer.lemmatize(w)
        cleaned_words.append(w)
    return ' '.join(cleaned_words)

def clean_meaningful_words(text):
    if pd.isna(text):
        return ""
    return ' '.join([w for w in text.split() if is_meaningful_word(w)])

# Load data
@st.cache_data
def load_data():
    # Try to load cleaned data if exists, else load raw and clean
    if os.path.exists('tweets_cleaned_final.csv'):
        df = pd.read_csv('tweets_cleaned_final.csv')
    else:
        df = pd.read_csv('tweet_sentiment.csv', encoding='latin-1')
        # Rename columns
        df = df.rename(columns={
            'tweet_text': 'Tweet',
            'emotion_in_tweet_is_directed_at': 'Target',
            'is_there_an_emotion_directed_at_a_brand_or_product': 'Sentiment'
        })
        # Clean
        df['Cleaned_Tweet'] = df['Tweet'].apply(clean_text_comprehensive)
        df = df[df['Cleaned_Tweet'].str.len() > 5].copy()
        df['Cleaned_Tweet_Final'] = df['Cleaned_Tweet'].apply(clean_meaningful_words)
        # Save for future
        df.to_csv('tweets_cleaned_final.csv', index=False)
    return df

# Train model and vectorizer
@st.cache_resource
def train_model():
    df = load_data()
    # Filter out "I can't tell"
    df_clean = df[df['Sentiment'] != "I can't tell"].copy()
    df_clean['Sentiment'] = df_clean['Sentiment'].replace({
        'Positive emotion': 'Positive',
        'Negative emotion': 'Negative',
        'No emotion toward brand or product': 'Neutral'
    })
    le = LabelEncoder()
    df_clean['label'] = le.fit_transform(df_clean['Sentiment'])
    
    X = df_clean['Cleaned_Tweet_Final']
    y = df_clean['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Use best model from notebook: LinearSVC with balanced class_weight, C=0.1
    svm_model = LinearSVC(class_weight='balanced', C=0.1, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    
    # Save model and vectorizer
    with open('model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    
    return svm_model, tfidf, le

@st.cache_resource
def load_model():
    if os.path.exists('model.pkl') and os.path.exists('vectorizer.pkl') and os.path.exists('label_encoder.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, vectorizer, le
    else:
        return train_model()

# Prediction function
def predict_sentiment(text, model, vectorizer, le):
    cleaned = clean_text_comprehensive(text)
    cleaned = clean_meaningful_words(cleaned)
    if not cleaned:
        return "Empty or invalid input"
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    return le.inverse_transform([pred])[0]

# Main Streamlit app
def main():
    st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="wide")
    st.title("📊 Tweet Sentiment Analysis")
    st.markdown("Analyze sentiment in tweets about brands and products (Apple, Google, etc.)")

    # Load data and model
    df = load_data()
    model, vectorizer, le = load_model()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose a section", 
        ["Predict Sentiment", "Data Overview", "Sentiment Distribution", "Brand Analysis"])

    if app_mode == "Predict Sentiment":
        st.header("🔮 Predict Sentiment of a Tweet")
        tweet_input = st.text_area("Enter a tweet:", height=150)
        if st.button("Predict"):
            if tweet_input.strip():
                sentiment = predict_sentiment(tweet_input, model, vectorizer, le)
                st.success(f"Predicted Sentiment: **{sentiment}**")
            else:
                st.warning("Please enter a tweet.")

        st.subheader("Example Tweets")
        examples = [
            "I absolutely love my new iPhone! Best phone ever.",
            "The battery life on my Google Pixel is terrible, it dies in hours.",
            "Just got the iPad Pro, it's amazing for productivity.",
            "Google's new update broke my app, very frustrating.",
            "Neutral tweet about tech stuff."
        ]
        for ex in examples:
            if st.button(f"Try: {ex[:50]}..."):
                sentiment = predict_sentiment(ex, model, vectorizer, le)
                st.info(f"Tweet: {ex}\n\nPredicted Sentiment: **{sentiment}**")

    elif app_mode == "Data Overview":
        st.header("📋 Dataset Overview")
        st.write(f"Total tweets: {len(df)}")
        st.write("Columns:", df.columns.tolist())
        st.write("Sample data:")
        st.dataframe(df[['Tweet', 'Sentiment', 'Target']].head(10))

    elif app_mode == "Sentiment Distribution":
        st.header("📊 Sentiment Distribution")
        # Clean sentiment labels
        sentiment_map = {
            'No emotion toward brand or product': 'Neutral',
            'Positive emotion': 'Positive',
            'Negative emotion': 'Negative',
            "I can't tell": 'Uncertain'
        }
        df['Sentiment_Cleaned'] = df['Sentiment'].map(sentiment_map)
        counts = df['Sentiment_Cleaned'].value_counts()
        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(counts.index, counts.values, color=['#808080','#2ca02c','#d62728','#1f77b4'])
        ax.set_title('Sentiment Distribution')
        ax.set_xlabel('Sentiment')
        ax.set_ylabel('Number of Tweets')
        for i, (label, count) in enumerate(counts.items()):
            ax.text(i, count+10, str(count), ha='center')
        st.pyplot(fig)

        # Also show percentages
        st.write("Percentage breakdown:")
        st.dataframe(pd.DataFrame({
            'Sentiment': counts.index,
            'Count': counts.values,
            'Percentage': (counts.values/len(df)*100).round(1)
        }))

    elif app_mode == "Brand Analysis":
        st.header("🏷️ Brand Analysis")
        # Map targets to brand groups
        brand_map = {
            'Apple': 'Apple', 'iPad': 'Apple', 'iPhone': 'Apple', 'Apple App': 'Apple',
            'Google': 'Google', 'Android': 'Google'
        }
        df['Brand'] = df['Target'].map(brand_map).fillna('Other')
        # Filter to only Apple and Google for clarity
        brand_df = df[df['Brand'].isin(['Apple', 'Google'])]
        if not brand_df.empty:
            ct = pd.crosstab(brand_df['Brand'], brand_df['Sentiment_Cleaned'])
            fig, ax = plt.subplots(figsize=(10,6))
            ct.plot(kind='bar', ax=ax, color=['#808080','#2ca02c','#d62728','#1f77b4'])
            ax.set_title('Sentiment Distribution by Brand')
            ax.set_xlabel('Brand')
            ax.set_ylabel('Count')
            ax.legend(title='Sentiment')
            for p in ax.patches:
                ax.annotate(str(p.get_height()), (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', fontsize=9)
            st.pyplot(fig)
        else:
            st.warning("No Apple or Google brand data found.")

        # Also show top negative phrases (simplified)
        st.subheader("Top Negative Bigrams (Apple vs Google)")
        # We'll just show a quick word frequency of negative tweets
        neg_df = df[df['Sentiment_Cleaned'] == 'Negative']
        if not neg_df.empty:
            from collections import Counter
            import re
            words = []
            for tweet in neg_df['Cleaned_Tweet_Final'].dropna():
                words.extend(tweet.split())
            word_counts = Counter(words)
            common_words = word_counts.most_common(10)
            st.write("Most common words in negative tweets:")
            st.write(common_words)
        else:
            st.write("No negative tweets found.")

if __name__ == "__main__":
    main()