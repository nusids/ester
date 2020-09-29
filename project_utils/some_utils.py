import scipy
import numpy as np
import pandas as pd
import pickle

import time

import nltk
from nltk.tokenize.casual import TweetTokenizer

import os

def tokenize_text(text, stopwords = list(nltk.corpus.stopwords.words('english'))):
   return [w.lower() for w in TweetTokenizer().tokenize(text) if len(w) > 1 and w not in stopwords]


def get_cooc_matrix(df_column, edge_threshold = 1):
    vocabulary = {}
    data = []
    row = []
    col = []
    counter = {}

    for i, doc in enumerate(df_column):
        for term in doc:
            counter[term] = counter.get(term, 0) + 1
    
    for i, doc in enumerate(df_column):
        for term in doc:
            if counter[term] >= edge_threshold:
                j = vocabulary.setdefault(term, len(vocabulary))
                data.append(1)
                row.append(i)
                col.append(j)

    M = scipy.sparse.coo_matrix((data, (row, col)))
    
    M = (M.T*M).tolil()
    M.setdiag([0]*len(vocabulary), k=0)
    M = M.tocsr()
   
    row, col = np.nonzero(M >= edge_threshold)
    mapping =  {old_id: i for i, old_id in enumerate(list(set(row)))}

    new_data = np.asarray(M[row, col]).ravel()
    new_row = [mapping[i] for i in row]
    new_col = [mapping[i] for i in col]
    M = scipy.sparse.coo_matrix((new_data, (new_row, new_col)))
    vocabulary = {w: mapping[old_id] for w, old_id in vocabulary.items() if old_id in mapping}
    
    return M, vocabulary

def column_normalize(M, norm='l1'):
    if norm == 'l1':
        colsum = np.array(M.sum(axis=0)).ravel()
        colsum[colsum == 0] = 1.
        return M.dot(scipy.sparse.diags(1./colsum))
    elif norm == 'l2':
        colsum = np.array(M.power(2).sum(axis=0)).ravel()
        colsum[colsum == 0] = 1.
        return M.dot(scipy.sparse.diags(1./np.sqrt(colsum)))
    raise ValueError("unsupported norm")
    
def row_normalize(M, norm='l1'):
    if norm == 'l1':
        rowsum = np.array(M.sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.
        return (scipy.sparse.diags(1./rowsum)).dot(M)
    elif norm == 'l2':
        rowsum = np.array(M.power(2).sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.
        return (scipy.sparse.diags(1./np.sqrt(rowsum))).dot(M)
    raise ValueError("unsupported norm")

def get_my_PR(M, p, alpha = 0.85, donormalize = True):
    if donormalize:
        M = column_normalize(M)
        p = p/sum(p)
    GM = alpha * M + (1 - alpha) * np.outer(p, np.ones(M.shape[0]))
    val, vec = scipy.sparse.linalg.eigs(GM, k=1)
    
    vec = np.abs(vec.flatten())
    return vec/sum(vec)
