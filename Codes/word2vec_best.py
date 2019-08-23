# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from gensim.models import Word2Vec
from functions import *

#################################################################
#################################################################
#################################################################

# Load pickle files

healthcare = openPickle('healthcare.pkl')
realestate = openPickle('realestate.pkl')
banking = openPickle('banking.pkl')
energy = openPickle('energy.pkl')

N=4 # no. of clusters

# Use filenames as labels
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

# Init stopwords
stopwords = getStopWords('stopwords.txt')

# Get corpus
corpus, flat = getCorpus([healthcare,realestate,banking,energy], stopwords)

#################################################################
#################################################################
#################################################################

# Get top k words

k=350
topN=200
    
top_k_words = getTopWords(flat,k)

filtered_corpus = getFilteredCorpus(corpus, top_k_words)

# Train WOrd2Vec
vecDim = 64
model_emb = Word2Vec(filtered_corpus, size=vecDim, window=8, min_count=1, negative=5, workers=4)

hc = model_emb.most_similar(['healthcare'], topn=topN)
hc_words = [w[0] for w in hc]

re = model_emb.most_similar(['real','estate'], topn=topN)
re_words = [w[0] for w in re]

ba = model_emb.most_similar(['banking'], topn=topN)
ba_words = [w[0] for w in ba]

en = model_emb.most_similar(['energy'], topn=topN)
en_words = [w[0] for w in en]

filter_words = []

filter_words += hc_words
filter_words += re_words
filter_words += ba_words
filter_words += en_words

filtered_corpus_w2v = getFilteredCorpus(corpus, filter_words)

filtered_corpus_w2v_string = [' '.join(report) for report in filtered_corpus_w2v]

# Convert corpus to TFIDF
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=0.4)
tfidf = vectorizer.fit_transform(filtered_corpus_w2v_string)
model = KMeans(n_clusters=N, max_iter=2000)

#################################################################
####################CLUSTER######################################
#################################################################

all_purity = []

for i in range(10):
    
    model.fit(tfidf)
         
    purity, groups = getPurity(labels, model.labels_, labels_files)
    
    all_purity.append(purity)

print('k: {}\ttopN: {}\tPurity: {:.2f}\tStdev: {:.2f}'.format(k,topN, np.mean(all_purity), np.std(all_purity)))


#################################################################
###################VISUALIZE#####################################
#################################################################

color = getColorLabel(labels)

tsne = visualize(tfidf.toarray(), color)

#####

topKwords=[]

for tfidf in tfidf.toarray():
    
    temp = getReviewTopWords(7, tfidf, vectorizer.vocabulary_)
    
    topKwords.append(temp)
    
###

table = makeTable(labels, labels_files, model.labels_, tsne, color, topKwords)

table.to_excel('word2vec.xlsx')