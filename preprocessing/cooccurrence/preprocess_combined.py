import sys, os
config_path = '../..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

import some_utils as ut

import pandas as pd

path = config_values['COOCCURRENCES']

wiki_cooc_file = os.path.join(path, 'wikigraph', 'wiki_cooccurrence.txt')
wiki_voc_file = os.path.join(path, 'wikigraph', 'wiki_vocabulary.txt')

twitter_cooc_file = os.path.join(path, 'twittergraph', 'twitter_cooccurrence.txt')
twitter_voc_file = os.path.join(path, 'twittergraph', 'twitter_vocabulary.txt')

comb_cooc_file = os.path.join(path, 'combgraph', 'comb_cooccurrence.txt')
comb_voc_file = os.path.join(path, 'combgraph', 'comb_vocabulary.txt')


wiki_voc = pd.read_csv(wiki_voc_file, header=None, sep=' ', names =['word'], keep_default_na = False).word.values
twitter_voc = pd.read_csv(twitter_voc_file, sep=' ', header=None, names =['word'], keep_default_na = False).word.values

use_from_twitter = [i for i,w in enumerate(twitter_voc) if w not in wiki_voc]
mapping = {old_id: new_id + len(wiki_voc) for new_id, old_id in enumerate(use_from_twitter)}

wiki_cooc = pd.read_csv(wiki_cooc_file, header=None, sep=' ', names =['row', 'col', 'data'])
twitter_cooc = pd.read_csv(twitter_cooc_file, header=None, sep=' ', names =['row', 'col', 'data'])

wiki_indx = {w: wiki_id for wiki_id, w in enumerate(wiki_voc)}

intersection_indx = [i for i,w in enumerate(twitter_voc) if w in wiki_voc]
mapping_intersection = {old_id: wiki_indx[twitter_voc[old_id]] for old_id in intersection_indx}

mapping.update(mapping_intersection)

tmp = twitter_cooc[twitter_cooc.row.isin(list(mapping.keys())) & twitter_cooc.col.isin(list(mapping.keys()))]
tmp['row'] = tmp['row'].map(mapping)
tmp['col'] = tmp['col'].map(mapping)
comb_cooc = wiki_cooc.append(tmp, ignore_index=True)
comb_voc = list(wiki_voc) + [twitter_voc[i] for i in use_from_twitter]
comb_cooc = comb_cooc.groupby(['row','col'])['data'].sum().reset_index()

comb_cooc.to_csv(comb_cooc_file, header = False, index = False, sep = ' ')
pd.DataFrame({'word': comb_voc}).to_csv(comb_voc_file, header = False, index = False, sep = ' ')