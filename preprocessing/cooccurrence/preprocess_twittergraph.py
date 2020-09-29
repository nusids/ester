import sys, os

config_path = '../..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

import some_utils as ut
import pandas as pd



cooc_file = os.path.join(config_values['COOCCURRENCES'], 'twittergraph', 'twitter_cooccurrence.txt')
voc_file = os.path.join(config_values['COOCCURRENCES'], 'twittergraph', 'twitter_vocabulary.txt')

df =  pd.read_csv(os.path.join(config_values['RAW_DATA'], 'sentiment140', 'training.1600000.processed.noemoticon.csv'),\
    quotechar = '"', sep = ',', engine='python', header=None, usecols=[5], names =['text'])
df.dropna(inplace=True)

df['tokenized'] = df.text.apply(lambda x: ut.tokenize_text(x))

M, vocabulary = ut.get_cooc_matrix(df.tokenized, edge_threshold = 10)
pd.DataFrame({'row': M.row, 'col': M.col, 'data': M.data}).to_csv(cooc_file, header = False, index = False, sep = ' ')
pd.DataFrame({'word': [k for k, v in sorted(vocabulary.items(), key=lambda item: item[1])]}).to_csv(voc_file, header = False, index = False, sep = ' ')

