# Twitter Sentiment Analysis - Naive Bayes Implementation

This notebook implements sentiment analysis on Twitter data using Naive Bayes classification.

## Overview

The project includes:
- A Naive Bayes classifier for tweet sentiment analysis
- Tweet preprocessing and tokenization
- Model training and evaluation
- Word sentiment analysis utilities

## Requirements

- Python 3.x
- NLTK
- NumPy
- Pandas

## Setup

```python
# Import required libraries
import nltk
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('twitter_samples')
```

## Data Preparation

The dataset consists of:
- Training data: 8000 tweets (4000 positive, 4000 negative)
- Test data: Remaining tweets from Twitter samples
- Binary labels (1 for positive, 0 for negative)

## Main Functions

### Tweet Processing
```python
custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"
processed_tweet = process_tweet(custom_tweet)
```

### Frequency Counting
```python
freqs = count_tweets(tweets, ys)
```

### Model Training
```python
logprior, loglikelihood = train_naive_bayes(x_train, y_train, freqs)
```

### Prediction
```python
prediction = naive_bayes_predict(tweet, loglikelihood, logprior)
```

### Model Evaluation
```python
accuracy = test_naive_bayes(x_test, y_test, loglikelihood, logprior)
```

## Additional Features

### Word Sentiment Ratio
```python
ratio = get_ratio(freqs, word)
```

### Threshold-based Word Filtering
```python
word_list = get_words_by_threshold(freqs, label, threshold)
```

## Model Details

The implementation includes:
- Laplace smoothing for handling unseen words
- Log probability calculations to prevent underflow
- Word frequency analysis for sentiment strength

## Usage Example

```python
# Process a custom tweet
custom_tweet = "I am happy"
processed = process_tweet(custom_tweet)

# Make prediction
result = naive_bayes_predict(custom_tweet, loglikelihood, logprior)
print(f'{custom_tweet} -> {result:.2f}')
```
