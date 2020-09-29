import load
import some_utils
import numpy as np
import pandas as pd

import config_parser as cp

import algorithms as algs
import metrics

import argparse
import os
import pickle as pkl


config_values = cp.parse()

datadir = config_values['DOCS']
outdir = config_values['OUTPUT']
lexicondir = config_values['LEXICONS']
use_emotions = config_values['USE_EMOTIONS']

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d', '--dataset', type=str, help='datasets: semval2018, crowdflower, ssec, tec, dens.', default='semval2018')
    parser.add_argument('-l', '--lexicon', type=str, help='lexicons: NRC-EmoLex, DepecheMood.', default="NRC-EmoLex")
    parser.add_argument('-c', '--cooc', type=str, help='cooc matrices: wiki, twitter, comb.', default="comb")
    parser.add_argument('-e', '--embedding', type=str, help='embeddings: emo2vec100, ewe-uni300, sswe-u50.', default="sswe-u50")
    parser.add_argument('-m', '--method', type=str, help='methods: ester, ester_lexnorm, ester_lex2sent, cos_embedding, cos_lexicon.', default="ester")
    parser.add_argument('--output', '-o', type=str, help='file name to save the result score. \
        If specified, then the result is stored to OUTPUT directory from config.ini file.', default="")
    
    args = parser.parse_args()

    if args.method == 'ester':
        M, use_vocab_df, reverse_use_vocab = load.get_cooc_matrix(name = args.cooc)
        S, df, loc_emotions = load.get_docs(reverse_use_vocab, name = args.dataset)
        C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions = loc_emotions, name = args.lexicon)
        res = algs.ester(S, C, M)
        print('\n{0:-^30}\n'.format('parameters'))
        print('{:10} {:10}'.format('dataset:', args.dataset))
        print('{:10} {:10}'.format('lexicon:', args.lexicon))
        print('{:10} {:10}'.format('method:', args.method))
        print('{:10} {:10}'.format('coocurrence:', args.cooc))


    elif args.method == 'ester_lexnorm':
        M, use_vocab_df, reverse_use_vocab = load.get_cooc_matrix(name = args.cooc)
        S, df, loc_emotions = load.get_docs(reverse_use_vocab, name = args.dataset)
        C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions = loc_emotions, name = args.lexicon)
        res = algs.ester(S, C, M, lexnorm = True)
        print('\n{0:-^30}\n'.format('parameters'))
        print('{:10} {:10}'.format('dataset:', args.dataset))
        print('{:10} {:10}'.format('lexicon:', args.lexicon))
        print('{:10} {:10}'.format('method:', args.method))
        print('{:10} {:10}'.format('coocurrence:', args.cooc))

    elif args.method == 'ester_lex2sent':
        M, use_vocab_df, reverse_use_vocab = load.get_cooc_matrix(name = args.cooc)
        S, df, loc_emotions = load.get_docs(reverse_use_vocab, name = args.dataset)
        C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions = loc_emotions, name = args.lexicon)
        res = algs.ester(S, C, M, mode='emo2sent', lexnorm = True)
        print('\n{0:-^30}\n'.format('parameters'))
        print('{:10} {:10}'.format('dataset:', args.dataset))
        print('{:10} {:10}'.format('lexicon:', args.lexicon))
        print('{:10} {:10}'.format('method:', args.method))
        print('{:10} {:10}'.format('coocurrence:', args.cooc))

    elif args.method == 'cos_embedding':
        M, use_vocab_df, reverse_use_vocab = load.get_embedding(name = args.embedding)
        S, df, loc_emotions = load.get_docs(reverse_use_vocab, name = args.dataset)
        C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions = loc_emotions, name = args.lexicon)
        res = algs.cos_embedding(S, C, M)
        print('{0:-^30}\n'.format('parameters'))
        print('{:10} {:10}'.format('dataset:', args.dataset))
        print('{:10} {:10}'.format('lexicon:', args.lexicon))
        print('{:10} {:10}'.format('method:', args.method))
        print('{:10} {:10}'.format('embedding:', args.embedding))

    elif args.method == 'cos_lexicon':
        M, use_vocab_df, reverse_use_vocab = load.get_cooc_matrix(name = args.cooc)
        S, df, loc_emotions = load.get_docs(reverse_use_vocab, name = args.dataset)
        C, lexicon, loc_emotions = load.get_lexicon(use_vocab_df, use_emotions = loc_emotions, name = args.lexicon)
        res = algs.cos_lexicon(S, C)
        print('\n{0:-^30}\n'.format('parameters'))
        print('{:10} {:10}'.format('dataset:', args.dataset))
        print('{:10} {:10}'.format('lexicon:', args.lexicon))
        print('{:10} {:10}'.format('method:', args.method))



    dfpred = pd.DataFrame(res, columns = loc_emotions)
    dfpred[['id', 'text']] = df[['id', 'text']].copy()

    
    print('\n{0:-^30}'.format('results'))
    
    binary_true, binary_pred = metrics.get_classification(df, dfpred, loc_emotions, k=1)
    J, F1, P, R = metrics.get_classification_quality(binary_true, binary_pred, loc_emotions)

    print('\n{:15} {:10.4}'.format('jaccard@1', J))
    print('{:15} {:10.4}'.format('F1@1', F1))
    print('{:15} {:10.4}'.format('precision@1', P))
    print('{:15} {:10.4}'.format('recall@1', R))

    binary_true, binary_pred = metrics.get_classification(df, dfpred, loc_emotions, k=2)
    J, F1, P, R = metrics.get_classification_quality(binary_true, binary_pred, loc_emotions)
    print('\n{:15} {:10.4}'.format('jaccard@2', J))
    print('{:15} {:10.4}'.format('F1@2', F1))
    print('{:15} {:10.4}'.format('precision@2', P))
    print('{:15} {:10.4}'.format('recall@2', R))

    if args.output:
        script_outdir = output
        if not os.path.exists(script_outdir):
            os.makedirs(script_outdir)
        dfpred.to_csv(os.path.join(script_outdir, args.output + '_predictions.pkl'), index = False, sep = '\t') 
    
    

