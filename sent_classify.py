import io
import os
import re
import nltk
import math
import string
import requests
import json
import pickle
import numpy as np
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname

from nltk.corpus import stopwords

# Additional sklearn imports for SVM/Bayes + Feature Selection
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_selection import SelectPercentile, f_classif

from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold, GridSearchCV

from sentiment_analyzer import SentimentIntensityAnalyzer
import sentiment_analyzer

if __name__ == "__main__":
    data_dir = './data'  # Setting your own file path here.

    x_filename = 'tweets.txt'
    y_filename = 'labels.txt'
    tweets = data_preprocessing(data_dir, x_filename, y_filename)

    print("Loading data...")
    x = np.array(tweets)
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        y = np.array([int(line.strip()) for line in f.readlines()])

    # Feature Selection
    # def get_tfidf_feats(x):
        # return TfidfVectorizer(analyzer='word', ngram_range=(1,2)).fit_transform(x[:, 1])
        # return TfidfVectorizer().fit_transform(x)

    senti_classifier = SentimentIntensityAnalyzer()
    def get_senti_features(x):
        scores = [list(senti_classifier.polarity_scores(instance).values()) for instance in x]
        for score in scores:
            score[-1] += 1  # normalize to 0, 2 scale
        return np.array(scores)

    def get_rts_counts(x):
        scaler = MinMaxScaler()
        return scaler.fit_transform(x[:, 3].reshape(-1, 1))

    def get_len_tweet(x):
        return np.array([len(instance.split(' ')) for instance in x]).reshape(-1, 1)
    
    # Put features together
    feats_union = FeatureUnion([ 
        ('tfidf', TfidfVectorizer()),
        ('senti', FunctionTransformer(get_senti_features, validate=False)),
        # ('len', FunctionTransformer(get_len_tweet, validate=False))
        # ('rts', FunctionTransformer(get_rts_counts, validate=False))
    ])

    x_feats = feats_union.fit_transform(x)
    f_selector = SelectPercentile(f_classif, percentile=60)
    f_selector.fit(x_feats, y)
    x_feats = f_selector.transform(x_feats).toarray()
    print(x_feats.shape)

    classifier = SVC(C=1, gamma=1)
    model = 
    predicts = model.predict(x_feats)


def data_preprocessing(data_dir, x_filename):
    # load and process samples
    print('start loading and process samples...')

    # addition
    stops = set(stopwords.words('english'))
    stops.add('rt') # can remove

    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    with open(os.path.join(data_dir, x_filename), encoding='utf-8') as f:
        for i, line in enumerate(f):
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['text'].replace("\n", " ")

            postprocess_tweet = []
            words = preprocess(content)

            for word in words:
                if word not in stops:
                    postprocess_tweet.append(word)
                    if word in words_stat.keys():
                        words_stat[word][0] += 1
                        if i != words_stat[word][2]:
                            words_stat[word][1] += 1
                            words_stat[word][2] = i
                    else:
                        words_stat[word] = [1,1,i]
            tweets.append(' '.join(postprocess_tweet))

    print("Preprocessing is completed")
    return tweets


def rm_html_tags(str):
    html_prog = re.compile(r'<[^>]+>', re.S)
    return html_prog.sub('', str)


def rm_html_escape_characters(str):
    pattern_str = r'&quot;|&amp;|&lt;|&gt;|&nbsp;|&#34;|&#38;|&#60;|&#62;|&#160;|&#20284;|&#30524;|&#26684|&#43;|&#20540|&#23612;'
    escape_characters_prog = re.compile(pattern_str, re.S)
    return escape_characters_prog.sub('', str)


def rm_at_user(str):
    return re.sub(r'@[a-zA-Z_0-9]*', '', str)


def rm_url(str):
    return re.sub(r'http[s]?:[/+]?[a-zA-Z0-9_\.\/]*', '', str)


def rm_repeat_chars(str):
    return re.sub(r'(.)(\1){2,}', r'\1\1', str)


def rm_hashtag_symbol(str):
    return re.sub(r'#', '', str)


def rm_time(str):
    return re.sub(r'[0-9][0-9]:[0-9][0-9]', '', str)

def rm_punctuation(current_tweet):
    return re.sub(r'[^\w\s]','',current_tweet)

porter = nltk.PorterStemmer()

def preprocess(str):
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)

    try:
        str = nltk.tokenize.word_tokenize(str)
        try:
            str = [porter.stem(t) for t in str]
        except:
            print(str)
            pass
    except:
        print(str)
        pass
        
    return str