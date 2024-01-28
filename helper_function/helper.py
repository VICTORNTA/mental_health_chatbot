import re
import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pickle as pk #load pickle library to get tokenizer file
import nltk #load nltk library for nlp task
from keras.preprocessing.text import Tokenizer #import tokenizer
from keras.preprocessing.sequence import pad_sequences #import pad_sequence to padd tweets/text
import joblib as jb
import os

nltk.download(['punkt','stopwords']) #download the diffrent stopwords available on nltk

stop_words = set(stopwords.words('english'))

dep_detec = jb.load('./models/first_model.jb') #loading model

os.chdir('./helper_function')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pk.load(handle)

def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean

def process_multiple_tweets(tweets):

    MAX_SEQUENCE_LENGTH = 2495 #pad each tweet to 2495 tokens

    processed_tweets = [process_tweet(tweet) for tweet in tweets] #process the each tweet/text in the function

    tweets_non_stopwords = [[word for word in tweet_words if not word in stop_words] #check for stopwords and remove them
              for tweet_words in processed_tweets]

    #check for more stopwords and remove them
    collection_words = ['im', 'de', 'like', 'one']
    tweets_non_stopwords2 = [[w for w in word if not w in collection_words]
                 for word in tweets_non_stopwords]
    #Tokenize the processed tweets/texts
    text_to_seq = tokenizer.texts_to_sequences(tweets_non_stopwords2)
    pad_token_tweets = pad_sequences(text_to_seq,maxlen=MAX_SEQUENCE_LENGTH) #pad each tweets to have a length of 2495 which was the maximum length of tweets used during training
    return pad_token_tweets

def single_tweet_process(tweet):
    MAX_SEQUENCE_LENGTH = 2495

    processed_tweet = process_tweet(tweet)

    tweets_non_stopwords = [word for word in processed_tweet]

    collection_words = ['im', 'de', 'like', 'one']
    tweets_non_stopwords2 = [[w for w in tweets_non_stopwords if not w in collection_words]]
    text_to_seq = tokenizer.texts_to_sequences(tweets_non_stopwords2)

    pad_token_tweets = pad_sequences(text_to_seq,maxlen=MAX_SEQUENCE_LENGTH)
    return pad_token_tweets

#Function to predict tweets/tweet
def predict_tweet(tweets):
    if len(tweets) == 1:
      #if the user just provides one tweet pass it to the single tweet function
        only_tweet = single_tweet_process(tweets[0])
        #predict the tweet using the model
        pred = dep_detec.predict(only_tweet)
        #if the prediction is greater than 0.5 the person is depressive
        if pred[0][0] > 0.5:
            print(pred[0][0])
            print('Based On your Social media appearance you are depressive')
        else:
          #if the prediction is less than 0.5 the person is not depressive
            print('Based On your Social media appearance you are not depressive')
    elif len(tweets) > 1:
      #if the user just provides one tweet pass it to the single tweet function
        all_tweets = process_multiple_tweets(tweets)
        pred = dep_detec.predict(all_tweets)
        if pred.mean() > 0.5:
            return('Based on your social media appearance you sound depressive')
        else:
            return ('Based on your social media appearance you do notsound depressive')
