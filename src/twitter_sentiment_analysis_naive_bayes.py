import nltk
from nltk.corpus import stopwords, twitter_samples
import numpy as np
import pandas as pd
from nltk.tokenize import TweetTokenizer

from utils_w2 import process_tweet, lookup

nltk.download('stopwords')
nltk.download('twitter_samples')

all_pos = twitter_samples.strings('positive_tweets.json')
all_neg = twitter_samples.strings('negative_tweets.json')

train_pos = all_pos[:4000]
test_pos = all_pos[4000:]

train_neg = all_neg[:4000]
test_neg = all_neg[4000:]

x_train = train_pos + train_neg
x_test = test_pos + test_neg

y_train = np.concatenate([np.ones((len(train_pos))) , np.zeros((len(train_neg)))], axis = 0)
y_test = np.concatenate([np.ones((len(test_pos))) , np.zeros((len(test_neg)))], axis = 0)

len(train_pos), len(test_pos), len(train_neg), len(test_neg), len(y_train), len(y_test)

custom_tweet = "RT @Twitter @chapagain Hello There! Have a great day. :) #good #morning http://chapagain.com.np"

process_tweet(custom_tweet)

def count_tweets(tweets, ys, res = {}):

  for (tweet, y) in zip(tweets, ys):
    for word in process_tweet(tweet):

      pair = (word, y)
      res[pair] = res.get(pair,0) +1

  return res

tweets = x_train
ys = y_train

freqs = count_tweets(tweets, ys)

len(freqs)

def train_naive_bayes(x, y, freqs):

  N_pos,N_neg, V_pos,V_neg = 0, 0, 0, 0

  for pair in freqs:

    # check the word pos or neg, get freq of each word in pos and neg
    if pair[1] > 0:
      N_pos += freqs[pair]
      V_pos += 1

    else:
      N_neg += freqs[pair]
      V_neg += 1

  # get total pos and neg words
  D_pos = (y == 1).sum()
  D_neg = (y == 0).sum()

  # calculate prior
  logPrior = np.log(D_pos / D_neg)

  # get set of all words "vocab"
  vocab = set([pair[0] for pair in freqs])
  V = len(vocab)

  logliklyhood = {}

  for word in vocab:

    # for each word in vocab, het the pos and neg count
    freq_pos = lookup(freqs, word, 1)
    freq_neg = lookup(freqs, word, 0)

    # get proba for for each word in a desired class
    p_w_pos = (freq_pos +1) / (N_pos + V)
    p_w_neg = (freq_neg +1) / (N_neg + V)

    logliklyhood[word] = np.log(p_w_pos / p_w_neg)

  return logPrior,logliklyhood

logprior, loglikelihood  = train_naive_bayes(x_train, y_train, freqs)

len(loglikelihood), logprior

def naive_bayes_predict(tweet, loglikelihood, logprior):

  # process tweet
  processed = process_tweet(tweet)
  proba = logprior

  for word in processed:
    proba += loglikelihood.get(word,0)

  return proba

tweet = 'she smiled.'
naive_bayes_predict(tweet,loglikelihood, logprior)

def test_naive_bayes(x_test, y_test, loglikelihood, logprior):

  pred = []
  for tweet in x_test:

    y_hat = naive_bayes_predict(tweet, loglikelihood, logprior)
    pred.append(1 if y_hat > 0 else 0)

  # return accuracy for y_test and pred
  return (y_test == pred).sum() / len(pred)

pred = test_naive_bayes(x_test, y_test,loglikelihood, logprior)

= ['I am happy', 'I am bad',
    'this movie should have been great.',
    'great', 'great great', 'great great great',
    'great great great great']

for tweet in test_tweets:

  p = naive_bayes_predict(tweet, loglikelihood, logprior)
  print(f'{tweet} -> {p:.2f}')

def get_ratio(freqs, word):

  pos_neg_ratio = {'positive' : 0, 'negative' : 0, 'ratio' : 0}

  pos_neg_ratio['positive'] = lookup(freqs, word, 1)
  pos_neg_ratio['negative'] = lookup(freqs, word, 0)

  pos_neg_ratio['ratio'] = (lookup(freqs, word, 1) +1) / (lookup(freqs, word, 0) +1)

  return pos_neg_ratio

get_ratio(freqs, 'happy')

def get_words_by_threshold(freqs, label, threshold):

 # get most n words in a specific class
 # ex, get the fifty most positive words
  word_list = {}
  for pair in freqs.keys():

    word,_ = pair
    pos_neg_ratio = get_ratio(freqs, word)

    if label == 1 and pos_neg_ratio['ratio'] >= threshold:
      word_list[word] = pos_neg_ratio

    elif label == 0 and pos_neg_ratio['ratio'] <= threshold:
      word_list[word] = pos_neg_ratio

  return word_list

get_words_by_threshold(freqs, 1, 50)
