# encoding=utf-8
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
# from sentiment_analyzer import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from itertools import product
from inspect import getsourcefile
from os.path import abspath, join, dirname

# SVM/Bayes imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

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


porter = nltk.PorterStemmer()

def pre_process(str):
    # do not change the preprocessing order only if you know what you're doing
    str = str.lower()
    str = rm_url(str)
    str = rm_at_user(str)
    str = rm_repeat_chars(str)
    str = rm_hashtag_symbol(str)
    str = rm_time(str)


    # additions
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


def data_preprocessing(data_dir, x_filename, y_filename):
    # load and process samples
    print('start loading and process samples...')

    # addition
    stops = set(stopwords.words('english'))
    stops.add('rt') # can remove

    words_stat = {}  # record statistics of the df and tf for each word; Form: {word:[tf, df, tweet index]}
    tweets = []
    cnt = 0
    with open(os.path.join(data_dir, x_filename), encoding='utf-8') as f:
        for i, line in enumerate(f):
            tweet_obj = json.loads(line.strip(), encoding='utf-8')
            content = tweet_obj['text'].replace("\n", " ")
            postprocess_tweet = pre_process(content)
            words = pre_process(content, porter)
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
            # tweets.append(postprocess_tweet)

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

    # Set up classifier
    # classifier = SentimentIntensityAnalyzer()
    param_grid = {
        'C' : [0.001, 0.01, 0.1, 1, 10],
        'gamma' : [0.001, 0.01, 0.1, 1]
    }
    classifier = GridSearchCV(SVC(), param_grid, cv=10)

    print("Loading data...")
    # with open(os.path.join(data_dir, 'tweets_processed.txt'), 'r') as f:
    #    x = np.array(f.readlines())

    x = tweets
    with open(os.path.join(data_dir, 'labels.txt'), 'r') as f:
        y = np.array([int(line.strip()) for line in f.readlines()])

    # Feature Selection
    x_feats = TfidfVectorizer().fit_transform(x)

    from sklearn.feature_selection import SelectPercentile, f_classif
    f_selector = SelectPercentile(f_classif, percentile=40)
    f_selector.fit(x_feats, y)
    x_feats = f_selector.transform(x_feats).toarray()
    print(x_feats.shape)

    print("Start training and predict...")
    kf = KFold(n_splits=10)
    avg_p_all, avg_p_pos, avg_p_neg, avg_p_net = 0, 0, 0, 0
    avg_r_all, avg_r_pos, avg_r_neg, avg_r_net = 0, 0, 0, 0
    for train, test in kf.split(y):

        model = classifier.fit(x[train], y[train])
        print('Curr best svm params: %s', classifier.best_params_, end='\n')
        predicts = model.predict(x[test])

        # predict_scores = []
        # for ind in test:
        #     # for instance in x[test]:
        #       instance = x[ind]
        #       predict_scores.append(classifier.polarity_scores(instance)["compound"])
        # predicts = sentiment_analyzer.map_to_label(predict_scores)
        # print(classification_report(y[test], predicts))
        
        # pos = label 2
        avg_p_pos += precision_score(y[test], predicts, labels=[2], average='macro')
        avg_r_pos += recall_score(y[test], predicts, labels=[2], average='macro')

        # neutral = label 1
        avg_p_net += precision_score(y[test], predicts, labels=[1], average='macro')
        avg_r_net += recall_score(y[test], predicts, labels=[1], average='macro')

        # negative = label 0
        avg_p_neg += precision_score(y[test], predicts, labels=[0], average='macro')
        avg_r_neg += recall_score(y[test], predicts, labels=[0], average='macro')

        avg_p_all += precision_score(y[test], predicts, average='macro')
        avg_r_all += recall_score(y[test], predicts, average='macro')

    print('Average Positive Precision is %f' % (avg_p_pos / 10.0))
    print('Average Positive Recall is %f' % (avg_r_pos / 10.0), end='\n')
    print('--------------------', end='\n')
    print('Average Neutral Precision is %f' % (avg_p_net / 10.0))
    print('Average Neutral Recall is %f' % (avg_r_net / 10.0), end='\n')
    print('--------------------', end='\n')
    print('Average Negative Precision is %f' % (avg_p_neg / 10.0))
    print('Average Negative Recall is %f' % (avg_r_neg / 10.0), end='\n')
    print('--------------------', end='\n')
    print('Average Total Precision is %f' % (avg_p_all / 10.0))
    print('Average Total Recall is %f' % (avg_r_all / 10.0), end='\n')
