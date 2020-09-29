import os
import config_parser as cp
config_values = cp.parse(os.path.dirname(os.path.realpath(__file__)))

import pandas as pd
import numpy as np
import sklearn
import scipy
from gensim.models import KeyedVectors

import some_utils as ut
import zipfile


def get_cooc_matrix(name = 'toy'):
    filepath = os.path.join(config_values['COOCCURRENCES'], name+'graph')

    cooc_matrix_df = pd.read_csv(os.path.join(filepath, name + '_cooccurrence.txt'), sep = ' ', names = ['row', 'col', 'data'])
    M = scipy.sparse.coo_matrix((cooc_matrix_df.data.values, (cooc_matrix_df.row.values, cooc_matrix_df.col.values)))
    
    use_vocab_df = pd.read_csv(os.path.join(filepath, name + '_vocabulary.txt'), sep = ' ', names = ['word'], keep_default_na = False)
    use_vocab_df['index'] = list(range(len(use_vocab_df)))
    use_vocab_dict = {i: w for i, w in enumerate(use_vocab_df.word.values)}
    reverse_use_vocab = {w: i for i, w in use_vocab_dict.items()}
    return M, use_vocab_df, reverse_use_vocab

def get_embedding(name = 'toy'):
    path = config_values['EMBEDDINGS']
    file_path = os.path.join(path, name + '.txt')
    if not os.path.exists(file_path):
        path_to_zip_file = os.path.join(path, name + '.zip')
        with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
            zip_ref.extractall(path)
            
    model = KeyedVectors.load_word2vec_format(file_path)

    E = np.empty((len(model.wv.vocab), model.vector_size))
    reverse_use_vocab = {}
    for idx, key in enumerate(model.wv.vocab):
        E[idx, :] = model.wv[key]
        reverse_use_vocab[key] = idx
    use_vocab_df = pd.DataFrame({'word': list(model.wv.vocab.keys()), 'index': list(range(len(model.wv.vocab)))})

    return scipy.sparse.coo_matrix(E), use_vocab_df, reverse_use_vocab

def get_lexicon(use_vocab_df, use_emotions, out_sparse = True, name = 'toy'):
    filepath = config_values['LEXICONS']
    #use_emotions = config_values['USE_EMOTIONS']

    lex =  pd.read_csv(os.path.join(filepath, name + '.txt'), sep = '\t')
    projected_lexicon = use_vocab_df.set_index('word').join(lex.set_index('word'), on='word').reset_index()
    projected_lexicon.fillna(0, inplace = True)

    projected_lexicon['neutral'] = 0.
    loc_emotions = [e for e in use_emotions if e in list(projected_lexicon)]
    norm = projected_lexicon[loc_emotions].sum(axis = 1).copy()
    projected_lexicon.loc[norm == 0, ('neutral')] = 1.

    if sum(list(projected_lexicon.neutral.values)) == 0:
        projected_lexicon['neutral'] = 1.
   
    C = projected_lexicon[loc_emotions].values
    if out_sparse:
        C = scipy.sparse.csc_matrix(C)
    return C, projected_lexicon, loc_emotions

def get_docs(reverse_use_vocab, out_sparse = True, name = 'toy'):
    filepath = config_values['DOCS']
    use_emotions = config_values['USE_EMOTIONS']

    df =  pd.read_csv(os.path.join(filepath, name + '.txt'), sep = '\t')
    loc_emotions = [e for e in use_emotions if e in list(df)]
    df['tokenized'] = df.text.apply(lambda x: ut.tokenize_text(x))
    
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary= True, vocabulary = reverse_use_vocab)
    S = vectorizer.fit_transform(df.tokenized.apply(' '.join))
    if out_sparse:
        S = S.tocsc()
    else:
        S = S.toarray()
    return S, df, loc_emotions



