# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 12:26:31 2019

@author: kevin
"""
import pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from functions import *


#################################################################
#################################################################
#################################################################

# Load pickle files

healthcare = openPickle('healthcare.pkl')
realestate = openPickle('realestate.pkl')
banking = openPickle('banking.pkl')
energy = openPickle('energy.pkl')
#transport = openPickle('transport.pkl')

N=4 # no. of clusters

# Use filenames as labels
with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)
L = len(labels)
# Init stopwords
stopwords = getStopWords('stopwords.txt')

# Get corpus
corpus, flat = getCorpus([healthcare,realestate,banking,energy], stopwords)

#################################################################
#################################################################
#################################################################

#### Find top-k words ####

k=400

top_k_words = getTopWords(flat,k)

filtered_corpus = getFilteredCorpus(corpus, top_k_words)

corpus_txt = [' '.join(review) for review in filtered_corpus]
# Convert to TF 
vectorizer = CountVectorizer(ngram_range=(1,1))
tf = vectorizer.fit_transform(corpus_txt)
model = KMeans(n_clusters=N, max_iter=2000) 

#################################################################
####################CLUSTER######################################
#################################################################
all_purity = []

for i in range(10):
    
    model.fit(tf)
         
    purity, groups = getPurity(labels, model.labels_, labels_files)
    
    all_purity.append(purity)

print('k: {}\tPurity: {:.2f}\tStdev: {:.2f}'.format(k, np.mean(all_purity), np.std(all_purity)))

#################################################################
###################VISUALIZE#####################################
#################################################################

color = getColorLabel(labels)

tsne = visualize(tf.toarray(), color)

#####

topKwords=[]

for tf in tf.toarray():
    
    temp = getReviewTopWords(7, tf, vectorizer.vocabulary_)
    
    topKwords.append(temp)
    
###

table = makeTable(labels, labels_files, model.labels_, tsne, color, topKwords)
    
table.to_excel('TF.xlsx')