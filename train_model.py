# encoding=utf8
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

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
# nltk.download()
from nltk.corpus import stopwords
#import simplejson as json
from sentiment_analyzer import SentimentIntensityAnalyzer
from sklearn.model_selection import KFold
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname
from sklearn.metrics import classification_report, precision_score, recall_score
import sentiment_analyzer


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


def pre_process(str):
    # do not change the preprocessing order only if you know what you're doing
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)

    return str


def data_preprocessing(data_dir, x_filename, y_filename):
    # load and process samples
    print('start loading and process samples...')
    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding='utf-8') as f:
        for i, line in enumerate(f):
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['text'].replace("\n", " ")
            postprocess_tweet = pre_process(content)
            tweets.append(postprocess_tweet)

    print("Preprocessing is completed")
    return tweets

    # Re-process samples, filter low frequency words...
    #fout = open(os.path.join(data_dir, 'tweets_processed.txt'), 'w')
    # for tweet in tweets:
    #    fout.write('%s\n' % tweet)
    # fout.close()

    #print("Preprocessing is completed")


if __name__ == "__main__":
    data_dir = './data'  # Setting your own file path here.

    x_filename = 'tweets.txt'
    y_filename = 'labels.txt'
    tweets = data_preprocessing(data_dir, x_filename, y_filename)

    classifier = SentimentIntensityAnalyzer()

    print("Loading data...")
    # with open(os.path.join(data_dir, 'tweets_processed.txt'), 'r') as f:
    #    x = np.array(f.readlines())

    x = tweets
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        y = np.array([int(line.strip()) for line in f.readlines()])

    print("Start training and predict...")
    kf = KFold(n_splits=10)
    avg_p = 0
    avg_r = 0
    for train, test in kf.split(y):

        # you can use the train data to train your classifiers
        # new_classifier = new_classifiler_model.fit(x[train], y[train])
        # then apply to the test data as below

        predict_scores = []
        #print(x[:10])
        #print(test.shape)

        for ind in test:
            # for instance in x[test]:
            instance = x[ind]
            #print(instance)
            predict_scores.append(classifier.polarity_scores(instance)["compound"])
        predicts = sentiment_analyzer.map_to_label(predict_scores)

        print(classification_report(y[test], predicts))
        avg_p += precision_score(y[test], predicts, average='macro')
        avg_r += recall_score(y[test], predicts, average='macro')

    print('Average Precision is %f.' % (avg_p / 10.0))
    print('Average Recall is %f.' % (avg_r / 10.0))
