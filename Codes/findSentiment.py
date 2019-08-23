# -*- coding: utf-8 -*-

import pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.cluster import KMeans
from utils import *

#################################################################
#################################################################
#################################################################

# Load pickle files

healthcare = openPickle('healthcare.pkl')
realestate = openPickle('realestate.pkl')
banking = openPickle('banking.pkl')
energy = openPickle('energy.pkl')


with open('labels_files.pkl', 'rb') as f:
    labels_files = pickle.load(f)

# Get corpus
corpus, flat = getCorpus([healthcare,realestate,banking,energy], stopwords=[])

#################################################################
#################################################################
#################################################################

# Get key words
pos = ['margin', 'expansion', 'turnaround', 'upside', 'leader', 'acceleration', 'attractive', 'improve', 'catalyst', 'confidence', 'strong', 'efficiency', 'appreciate', 'underestimate', 'see','opportunity',
       ]

neg = ['sell', 'downgrade', 'pressures', 'challenges', 'inefficient', 'decelerating', 'losses', 'margin pressure', 'critical', 'downside', 'over-confident', 'over-bullish', 'underweight', 'short balance','caution']


sent_arr = np.zeros((len(corpus), 3))

for i, review in enumerate(corpus):
    
    for word in review:
        
        if word in pos:
            sent_arr[i,0] += 1
        if word in neg:
            sent_arr[i,1] += 1
    
    p = sent_arr[i,0]
    n = sent_arr[i,1]
    
        
    if p-n > 6:
        
        sent_arr[i,2] = 1
        
    else:
        sent_arr[i,2] = 0
        
        
argmax = np.argmax(sent_arr, axis=1)

            
results = pd.DataFrame(data =labels_files, columns = ['FileName'])
results[['POS','NEG','Remarks']] = pd.DataFrame(sent_arr)
    
results.to_excel('sentiment_forInputs.xlsx')  
        

