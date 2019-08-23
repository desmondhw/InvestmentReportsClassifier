#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from functions import *

# Load pickle files

healthcare = openPickle('healthcare.pkl')
realestate = openPickle('realestate.pkl')
banking = openPickle('banking.pkl')
energy = openPickle('energy.pkl')

##### GET FEATURE WORDS ###########

# Get key words
combined = list(set(['margin', 'expansion', 'turnaround', 'upside', 'acceleration', 'attractive', 'improve', 'catalyst', 
            'confidence', 'strong', 'efficiency', 'appreciate','headwinds', 'underestimate','downgrade', 
            'pressures', 'challenges', 'inefficient', 'decelerating', 'losses', 'pressure', 'critical', 'downside',
            'underweight','rerating', 'tailwind',
            'falling', 'declining']))

# Get corpus
corpus, flat = getCorpus([healthcare,realestate,banking,energy], [])

corpus = [[w for w in sub_review if w in set(combined)] for sub_review in corpus]

corpus_str = [' '.join(rev) for rev in corpus]

# Create vectorizer
vectorizer = CountVectorizer(ngram_range=(1,1))
tfidf = vectorizer.fit_transform(corpus_str)

# Get X
X = tfidf.toarray()

# Get Y
Y = pd.read_excel('sentiment_forInputs.xlsx')
Y = Y['Remarks'].values

#### Train test split #####
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.7, random_state=42)


#### Logistic Regresion #####
clf = LogisticRegression(random_state=0, penalty='l2').fit(X_train, y_train)

y_pred = clf.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))


print('\n',classification_report(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))

#View results
weights = list(clf.coef_[0])
tokens = vectorizer.get_feature_names()
mapping = pd.DataFrame(list(zip(tokens, weights)), columns=['Token','Weight'])
mapping.sort_values(by=['Weight'], inplace=True, ascending=False)
print(mapping)
