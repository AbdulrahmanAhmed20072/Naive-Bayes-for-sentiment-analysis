# Naive-Bayes-for-sentiment-analysis
This project implements a Naive Bayes classifier for sentiment analysis on Twitter data. It processes tweets, calculates word frequencies, and predicts sentiment (positive or negative) using probabilistic modeling. The project includes training and testing on a dataset of positive and negative tweets."

## Features
- **Tweet Processing**: Tokenization, stopword removal, and stemming.
- **Naive Bayes Classifier**: Trains a model to predict sentiment based on word frequencies.
- **Testing**: Evaluates the model's accuracy on a test dataset.
- **Word Analysis**: Provides tools to analyze the most positive and negative words.

## Installation
1. Clone the repository:
   ```bash
2.pip install nltk numpy pandas
3.import nltk
nltk.download('stopwords')
nltk.download('twitter_samples')
