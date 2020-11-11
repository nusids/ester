import sys, os
config_path = '..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

import pandas as pd
import scipy
import numpy as np

import sklearn
import load
import algorithms as algs
import some_utils as ut

import pickle

import matplotlib
import matplotlib.pyplot as plt


filepath = config_values['CASE_STUDY']
use_emotions = config_values['USE_EMOTIONS']

def draw_heatmap(dfpred):
	rows = []
	_ = dfpred.apply(lambda row: [rows.append(list(row[list(dfpred)[:-1]]) + [nn]) 
	                         for nn in row.hashtags], axis=1)
	df_new = pd.DataFrame(rows, columns=dfpred.columns)
	df_new[loc_emotions] = df_new[loc_emotions].gt(0).astype(int)
	tophtags = df_new['hashtags'].value_counts()[:40].index

	HM = []
	for h in tophtags:
	    t = (df_new[df_new.hashtags == h][loc_emotions].sum()/len(df_new[df_new.hashtags == h])).values
	    HM.append(t)
	HM = np.array(HM).T

	matplotlib.rc('xtick', labelsize=13) 
	matplotlib.rc('ytick', labelsize=13) 


	fig, ax = plt.subplots()
	im = ax.imshow(HM)

	ax.set_xticks(np.arange(len(tophtags)))
	ax.set_yticks(np.arange(len(loc_emotions)))
	ax.set_xticklabels(tophtags)
	ax.set_yticklabels(loc_emotions)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
	         rotation_mode="anchor")

	for i in range(len(loc_emotions)):
	    for j in range(len(tophtags)):
	        text = ax.text(j, i, str(HM[i][j])[1:4],
	                       ha="center", va="center", color="w")

	fig.tight_layout()
	plt.show()

def read_case_study(reverse_use_vocab):
    table = []
    with open(os.path.join(filepath, 'content.txt'), 'r') as f:
        for line in f:
            l = line.strip().split(' ')
            table.append([l[0], l[1], ' '.join(l[2:])])
    df = pd.DataFrame(table, columns = ['id', 'date', 'text'])
    df['tokenized'] = df.text.apply(lambda x: ut.tokenize_text(x))
    df['hashtags'] = df.tokenized.apply(lambda x: list(set(w for w in x if w[0]=='#')))

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary= True, vocabulary = reverse_use_vocab)
    S = vectorizer.fit_transform(df.tokenized.apply(' '.join))
    S = S.tocsc()
    return S, df
	   
if __name__ == "__main__":

	M, use_vocab_df, reverse_use_vocab = load.get_cooc_matrix(name = 'comb')
	S, df = read_case_study(reverse_use_vocab)
	C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions, name = 'NRC-EmoLex')

	res = algs.ester(S, C, M)
	#pickle.dump(res, open('casestudy_pagerank.pkl', "wb"))
	#res = pickle.load(open('casestudy_pagerank.pkl', "rb"))
	
	dfpred = pd.DataFrame(res, columns = loc_emotions)
	dfpred[['id', 'text', 'tokenized', 'hashtags']] = df[['id', 'text', 'tokenized', 'hashtags']].copy()

	k=2
	threshold = np.sort(dfpred[loc_emotions].values, axis = 1)[:,-k]
	binary_pred = dfpred[loc_emotions].ge(threshold, axis='rows').astype(int)
	binary_pred[binary_pred.sum(axis = 1) == len(use_emotions)] = 0
	dfpred[loc_emotions] = binary_pred*dfpred[loc_emotions]
	draw_heatmap(dfpred)


