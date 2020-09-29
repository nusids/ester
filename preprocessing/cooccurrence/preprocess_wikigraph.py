import sys, os

config_path = '../..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

import some_utils as ut
import pandas as pd

path = config_values['COOCCURRENCES']

cooc_file = os.path.join(path, 'wikigraph', 'wiki_cooccurrence.txt')
voc_file = os.path.join(path, 'wikigraph', 'wiki_vocabulary.txt')

input_voc_file = os.path.join(config_values['RAW_DATA'], 'wikidata', 'terms_tf100_df5.txt')
input_cooc_file = os.path.join(config_values['RAW_DATA'], 'wikidata', 'all.wiki.edges.win5_ec200.txt')


vocabulary_df = pd.read_csv(input_voc_file, sep = ' ', names = ['word', 'df', 'tf'], keep_default_na = False)
vocabulary = {w: i for i, w in enumerate(vocabulary_df.word.values)}

cooc_df = pd.read_csv(input_cooc_file, sep = ' ', names = ['row', 'col', 'data'])
cooc_df.drop(cooc_df[cooc_df.col == cooc_df.row].index, inplace = True)
cooc_df['row'] -= 1
cooc_df['col'] -= 1

used_words_ids = set(list(cooc_df.row.values) + list(cooc_df.col.values))
used_voc = [w for w, i in vocabulary.items() if i in used_words_ids]

mapping_ids = {old_id: new_id for new_id, old_id in enumerate(used_words_ids)}
cooc_df['row'] = cooc_df['row'].map(mapping_ids)
cooc_df['col'] = cooc_df['col'].map(mapping_ids)

rows = list(cooc_df.row) + list(cooc_df.col)
cols = list(cooc_df.col) + list(cooc_df.row)
datas = list(cooc_df.data) + list(cooc_df.data)
pd.DataFrame({'row': rows, 'col': cols, 'data': datas}).to_csv(cooc_file, header = False, index = False, sep = ' ')
pd.DataFrame({'word': used_voc}).to_csv(voc_file, header = False, index = False, sep = ' ')