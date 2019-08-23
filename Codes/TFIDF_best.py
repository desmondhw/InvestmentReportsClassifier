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

#### Find top-k words ####
k=250
        
top_k_words = getTopWords(flat,k)

filtered_corpus = getFilteredCorpus(corpus, top_k_words)

corpus_txt = [' '.join(review) for review in filtered_corpus]

df = 0.4
# Convert corpus to TFIDF
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=df)
tfidf = vectorizer.fit_transform(corpus_txt)
model = KMeans(n_clusters=N, max_iter=2000)
print()

#################################################################
####################CLUSTER######################################
#################################################################

all_purity = []

for i in range(10):
    
    model.fit(tfidf)
         
    purity, groups = getPurity(labels, model.labels_, labels_files)
    
    all_purity.append(purity)

print('k: {}\tDf: {}\tPurity: {:.2f}\tStdev: {:.2f}'.format(k,df, np.mean(all_purity), np.std(all_purity)))


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

table.to_excel('TFIDF.xlsx')
