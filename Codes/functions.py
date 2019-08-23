#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import nltk
import pickle
import csv
import re
import os
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
import copy

warnings.filterwarnings("ignore")

def getStopWords(fileName):
    '''Takes in a text file fileName in str 
    format and extracts all the stopWords'''
    lem = WordNetLemmatizer()
    StopWords = []

    with open(fileName, 'r') as sw:
        for line in sw:
            line = line.rstrip('\n')
            StopWords.append(line.lower())

    # Get set
    SW = list(set([lem.lemmatize(w) for w in StopWords]))
    print("No. of StopWords: ", len(SW))
    return SW


def openPickle(file_Name):
    '''Opens pickle file and returns the contents'''
    fileObject = open(file_Name, 'rb')
    contents = pickle.load(fileObject)
    fileObject.close()
    return contents

def getFileNames(dirList):
    files = []
    for sector in dirList:      
        for filename in os.listdir('PDF/' + sector):
            files.append(filename)
    print("No. of Files: ", len(files))
    
    return files

def getCorpus(sectorList, stopwords):
    '''Returns a corpus list-of-list, and a flat-corpus'''
    '''Stopword removal implemented'''
    corpus = []

    # Each sector is a list of list
    for sector in sectorList:
        corpus.extend(sector)
    corpus = [[w for w in sublist if w not in stopwords] for sublist in corpus]
    flat = [w for sublist in corpus for w in sublist]

    return corpus, flat

def getTopWords(flat, k):

    fdist = nltk.FreqDist(flat)

    most_common = fdist.most_common(k)

    most_common = [w[0] for w in most_common]

    return most_common


def getFilteredCorpus(corpus, FilterWords):

    FilteredCorpus = [[w for w in sublist if w in FilterWords]
                      for sublist in corpus]

    return FilteredCorpus


def getExternalCorpus():

    def load_corpus(filename):
        docs = []
        with open(filename, 'r', errors='ignore') as f:
            # use csv.reader to process CSV file
            csv_reader = csv.reader(f, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count > 0:
                    # only process from second line onwards
                    # row[11] is the headline, row[14] is the news body
                    docs.append((row[11] + ' ' + row[14]).lower())
                line_count += 1
        return docs

    # load corpus
    docs = load_corpus('Full-Economic-News-DFE-839861.csv')

    # remove </br>
    docs = [doc.replace('</br>', ' ') for doc in docs]

    # tokenization
    docs = [word_tokenize(doc) for doc in docs]

    # keep English words only
    # if a token consists of only English letters, we assume it's a word
    docs = [[token for token in doc if token.isalpha()] for doc in docs]

    return docs


def RenameFiles(sector):

    for i, filename in enumerate(os.listdir('PDF/' + sector)):

        new_name = sector + '_' + str(i) + '.pdf'
        destination = 'PDF/' + sector + '/' + new_name

        source = 'PDF/' + sector + '/' + str(filename)

        os.rename(source, destination)

    return i


def readPDFs(folder):

    lem = WordNetLemmatizer()

    filename_pattern = '.+\.pdf'
    my_corpus = PlaintextCorpusReader(folder, filename_pattern)
    list_of_files = my_corpus.fileids()

    corpus = []

    # Get list of file names
    for i, file in enumerate(list_of_files):

        fp = open(folder + file, 'rb')
        print(i)

        parser = PDFParser(fp)
        doc = PDFDocument()
        parser.set_document(doc)
        doc.set_parser(parser)
        doc.initialize('')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        laparams.char_margin = 1.0
        laparams.word_margin = 1.0
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        text = ''

        for page in doc.get_pages():
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    text += lt_obj.get_text()

        last = text.rfind('Reg AC')

        if last != -1:
            text = text[0:last]

        corpus.append(text.split())

    # Alphabets, Lower and Lemmatize
    docs1 = [[w.lower() for w in sub_doc] for sub_doc in corpus]

    docs2 = [[w for w in sub_doc if re.search(
        '^[a-z]+$', w)] for sub_doc in docs1]

    docs3 = [[w for w in sub_doc if len(w) > 3] for sub_doc in docs2]

    processed_docs = [[lem.lemmatize(w) for w in sub_doc] for sub_doc in docs3]

    # Return List of Lists
    return processed_docs


def savePickle(data, filename):

    # open the file for writing
    fileObject = open(filename, 'wb')

    # this writes the object a to the
    # file named 'testfile'
    pickle.dump(data, fileObject)

    # here we close the fileObject
    fileObject.close()


def getPurity(true_labels, cluster_labels, file_labels):

    true_labels = np.array(true_labels).reshape((len(true_labels), 1))
    file_labels = np.array(file_labels).reshape((len(file_labels), 1))
    cluster_labels = cluster_labels.reshape((len(true_labels), 1))

    groups = np.concatenate((cluster_labels, true_labels, file_labels), axis=1)

    # Get array of classes
    classes = np.unique(groups[:, 0])

    cluster_purity = []

    for clas in classes:

        temp = groups[groups[:, 0] == clas]

        # Get the number of each class present in the cluster
        _, counts = np.unique(temp[:, 1], return_counts=True)

        # Get the number of elements from the most frequently occuring class
        Mij = counts.max()

        cluster_purity.append(Mij)

    return np.sum(cluster_purity)/true_labels.shape[0], groups


def getColorLabel(label):

    uni = np.unique(label)

    label = np.array(label)

    color = ['black', 'red', 'blue', 'green', 'purple']

    for i, item in enumerate(uni):

        label = np.where(label == item, color[i], label)

    return list(label)


def visualize(arr, color_labels):

    X_embedded = TSNE(n_components=2).fit_transform(arr)

    print(X_embedded.shape)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=color_labels)

    plt.show()

    return X_embedded


def makeTable(labels, labels_files, model_labels, tsne, color, topKwords):

    L = len(labels)
    truth = np.array(labels).reshape(L, 1)
    truth_files = np.array(labels_files).reshape(L, 1)
    color = np.array(color).reshape(L, 1)
    model_labels = model_labels.reshape(L, 1)
    topKwords = np.array(topKwords).reshape(L, 1)

    table = np.concatenate(
        (truth, color, truth_files, model_labels, tsne, topKwords), axis=1)

    table = pd.DataFrame(table, columns=[
                         'Actual', 'Colour', 'FileName', 'Cluster', 'X', 'Y', 'TopWords'])

    return table


def getReviewTopWords(n, tfidf, mapdict):

    tfidf_lst = list(tfidf)

    tfidf_lst.sort(reverse=True)

    topN = tfidf_lst[:n]

    wordlist = []

    for score in topN:

        index = np.argwhere(tfidf == score)[0][0]

        for k, v in mapdict.items():

            if v == index:

                wordlist.append(k)

    return '-'.join(wordlist)


if __name__ == '__main__':

    pass

    '''
    sectors = ['Healthcare','Realestate','Banking','Energy','Transport']
    
    #for sector in sectors:
        
        #RenameFiles(sector)
    
    for sector in sectors:
        
        data = None

        data = readPDFs('PDF/' + sector +'/')
        
        savePickle(data, sector + '.pkl')
        '''
