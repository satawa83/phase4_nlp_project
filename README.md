# phase4_nlp_project
This project development and evaluate based Natural Language Processing Model for Sentiment Classification of brand and Products Emotions**

The goal is to understand:

To determine the extent to which tweet content reflects sentiment

To assess the distribution of sentiment across tweets based on the brands mentioned within the tweet text

To analyze the distribution of target subcategories represented within tweet content

To examine the influence of tweet content and sentiment on target categories

To analyze customer complaints expressed in tweets toward specific brand products



Dataset Structure

Column	                Description

Tweet	                Original raw tweet text
Target	                Brand or product mentioned
Sentiment	            Human-labeled sentiment category
Cleaned_Tweet	        Preprocessed tweet text
Cleaned_Tweet_Final	Final cleaned version used for modeling


# Project Visualizations & Insights

The extent to which tweet content reflects sentiment

![Sentiment Distribution](./sentiment_distribution.png)

Insight
The dominant takeaway is that 59.3% of the entire dataset (5,371 tweets) is categorized as Neutral.


Distribution of sentiment across tweets based on the brands mentioned within the tweet text
![Tweet Text Brand Sentiment](./tweet_text_brand_sentiment.png)
Insight
For both companies, the majority of the conversation is Neutral.Google leans more heavily into neutrality at 63.8%., Apple sits at 51.4%.Apple Negative Sentiment: 7.8%, Google Negative Sentiment: 5.4%


Distribution of target subcategories represented within tweet content

![Target Distribution in Tweets](./target_distribution_in_tweets.png)

Insight 
- The most striking feature of the dataset is the dominance of the "No specific brand" category, At 5,784 tweets, it accounts for 63.8% of the entire dataset


To examine the influence of tweet content and sentiment on target categories
![Sentiment Text Driving Target](./sentiment_text_driving_target.png)
Insight - 
- Positive sentiment is the primary driver for both brands, making up 80.6% of the Apple-labeled segment and 81.7% of the Google-labeled segment.
- Negative sentiment is the secondary driver, accounting for 16.3% of Apple labels and 15.0% of Google labels.


Customer complaints expressed in tweets toward specific brand products

![Negative Phrase Analysis](./negative_phrase_analysis.png)

Insight
APP CRASH" is Apple’s biggest individual complaint (21.4%), while ANDROID BUG" (23.5%) and "SLOW UPDATE" (19.8%) dominate over 43% of the entire negative conversation chart

# Project Visualizations & Insights










### 4. Brand-Specific Sentiment Breakdown
A comparative view of raw emotional intent (Positive, Negative, Neutral) when a specific brand name is explicitly mentioned.


---

### 5. Macro Sentiment Distribution
The bird's-eye view of total emotional polarity across all 9,064 analyzed tweets in the dataset.
![Sentiment Distribution](./sentiment_distribution.png)




3. Imbalanced Classification Problem

The dataset presents a clear imbalance:

Neutral >> Positive >> Negative

Recommended approaches:

Class weighting in ML models
Oversampling minority classes
Using F1-score instead of accuracy

4. NLP Readiness

The presence of:

Cleaned_Tweet_Final
Standardized sentiment labels

makes this dataset ready for:

TF-IDF modeling
BERT-based sentiment classification
Topic modeling (LDA, BERTopic)

Suggested Next Steps
Build sentiment classification model (Logistic Regression / BERT)
Perform topic modeling on cleaned tweets
Create brand-wise sentiment heatmaps
Deploy dashboard (Streamlit / Power BI)

Conclusion

This dataset provides a strong foundation for sentiment analysis in the tech domain, with clear brand focus and well-prepared cleaned text. While sentiment imbalance exists, it reflects real-world social media behavior, making it valuable for production-grade NLP systems.