import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer

# Running for the first time? Uncomment the following lines
# Then comment them out again
# import nltk
# nltk.download('vader_lexicon')

# Load movie reviews dataset
pos_reviews = pd.read_csv('positive_reviews.csv')
neg_reviews = pd.read_csv('negative_reviews.csv')

reviews_df = pd.concat([pos_reviews, neg_reviews], ignore_index=True)

reviews = reviews_df['review'].tolist()
sentiments = reviews_df['sentiment'].tolist()

# Preprocess data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
y = sentiments

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train and test a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))

# Sample sentiment analysis using SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sample_review = "This movie is amazing and heartwarming"
print("Sentiment analysis score:", sia.polarity_scores(sample_review))
