import sys, os

import pandas as pd
import numpy as np

import argparse

config_path = '../..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

datadir = config_values['DOCS_RAW']
outdir = config_values['DOCS']
use_emotions = config_values['USE_EMOTIONS']


def write_out(outdir, datafile, df):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    loc_emotions = [i for i in use_emotions if i in list(df)]
    df[['id', 'text'] + loc_emotions].to_csv(os.path.join(outdir, datafile +'.txt'), index = False, sep = '\t') 

def preprocess(dataset, subset = ''):
    if dataset == 'semval2018':
        datafolder =  'SemVal2018_E-c'
        if subset == '':
            datafile = "2018-E-c-En-all"
        elif subset == 'gold':
            datafile = "2018-E-c-En-test-gold"

        df = pd.read_csv(os.path.join(datadir, datafolder, datafile+'.txt'), sep = '\t')
        df.rename(columns={"Tweet": "text"}, inplace = True)
        df['id'] = list(range(len(df)))
        df['neutral'] = 0
        norm = df[use_emotions].sum(axis = 1).copy()
        df.loc[norm == 0, ('neutral')] = 1
        write_out(outdir, dataset+subset, df)
        print('processed {}'.format(dataset))

    elif dataset == 'crowdflower':
        datafolder =  dataset
        data_file = "text_emotion"
        df = pd.read_csv(os.path.join(datadir, datafolder, data_file + '.csv'), sep = ',')
        df = pd.pivot_table(df, columns='sentiment', index='content', values = 'author', aggfunc='count', fill_value=0)
        df.reset_index(inplace = True)
        df.columns.name = 'id'
        df.rename(columns={"content": "text"}, inplace = True)

        mapping = {
                'fear': ['worry'],\
                'surprise': ['surprise'],\
                'sadness': ['sadness'],\
                'disgust': ['boredom', 'empty', 'hate'],\
                'anger': ['anger'],\
                'joy': ['enthusiasm', 'fun', 'happiness', 'love', 'relief'],
                'neutral': ['neutral']}

        new_df = pd.DataFrame()
        new_df['text'] = df['text'].replace("\t", " ", regex=True)
        
        for k, v in mapping.items():
            new_df[k] = df[v].sum(axis = 1)
        norm = new_df[mapping.keys()].sum(axis = 1).copy()
            
        for k, v in mapping.items():
            new_df[k] /= norm
        new_df.loc[norm == 0, ('neutral')] = 1

        new_df[list(mapping.keys())] = (new_df[list(mapping.keys())] >= 0.5).astype(int)
        new_df.loc[new_df[mapping.keys()].sum(axis = 1) == 0, ('neutral')] = 1

        new_df['id'] = list(range(len(new_df)))
        
        write_out(outdir, dataset+subset, new_df)
        print('processed {}'.format(dataset))

    elif dataset == 'ssec':
        datafolder = 'ssec-aggregated'
        if subset == '':
            data_file = "all-combined-0.5"
        elif subset == 'gold':
            data_file = "test-combined-0.5"
            
        df = pd.read_csv(os.path.join(datadir, datafolder, data_file + '.csv'), na_values = '---', sep = '\t', header = None, index_col = None, names= ['anger', 'anticipation','disgust', 'fear', 'joy', 'sadness','surprise', 'trust', 'text'])
        df.fillna(0, inplace = True)
        

        df[list(df)[:-1]] = df[list(df)[:-1]].astype(bool).astype(int)
        df['neutral'] = 0
        norm = df[use_emotions].sum(axis = 1).copy()
        df.loc[norm == 0, ('neutral')] = 1

        df['id'] = list(range(len(df)))
        write_out(outdir, dataset + subset, df)
        print('processed {}'.format(dataset))

    elif dataset == 'tec':
        datafolder =  'TEC'
        data_file = "Jan9-2012-tweets-clean"
        df = pd.read_csv(os.path.join(datadir, datafolder, data_file + '.txt'), sep = '\t', header = None, names = ['id', 'text', 'emotion'])
        
        df['dummy'] = 1 
        df = pd.pivot_table(df, columns='emotion', index='text', values='dummy', aggfunc='count', fill_value=0)
        df = df.reset_index()
        df.rename(columns={":: anger": "anger", ":: disgust": "disgust", ":: fear": "fear", ":: joy": "joy", ":: sadness": "sadness", ":: surprise": "surprise"}, inplace = True)

        df['neutral'] = 0
        loc_emotions = list(set(list(df)) & set(use_emotions))
        norm = df[loc_emotions].sum(axis = 1).copy()
        for k in loc_emotions:
            df[k] /= norm
        df.loc[norm == 0, ('neutral')] = 1
        df[loc_emotions] = (df[loc_emotions] >= 0.5).astype(int)
        df.loc[df[loc_emotions].sum(axis = 1) == 0, ('neutral')] = 1
        df['id'] = list(range(len(df)))
        write_out(outdir, dataset + subset, df)
        print('processed {}'.format(dataset))

    elif dataset == 'dens':
        datafolder = 'DENS'
        data_file = "dens"
        df = pd.read_csv(os.path.join(datadir, datafolder, data_file + '.tsv'), sep = '\t')
        df = pd.pivot_table(df, columns='Label', index='Text', values = 'Agreements', aggfunc='sum', fill_value=0)
        df.reset_index(inplace = True)
        df.columns.name = 'id'
        df.columns = map(str.lower, df.columns)
        df.rename(columns={"sad": "sadness", "love": "trust"}, inplace = True)
        df[use_emotions] = (df[use_emotions] >= 0.5).astype(int)
        df['id'] = list(range(len(df)))
        write_out(outdir, dataset+subset, df)
        print('processed {}'.format(dataset))

    else:
        raise ValueError('Unknown dataset name, run -h for available options.')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Preprocess datasets in format tsv with text field \"text\" and (if present) 8 Plutchik.')
    parser.add_argument('-d', '--dataset', type=str, help='datasets: semval2018, crowdflower, ssec, tec. Use \'all\' to process all datasets.', default='all')
    parser.add_argument('--gold', help='use gold datase partition if exists', action="store_false")
    args = parser.parse_args()
    dataset = args.dataset
    subset = 'gold' if args.gold else ''

    if dataset == 'all':
        all_datasets = ['semval2018', 'crowdflower', 'ssec', 'tec']
        for dataset in all_datasets:
            preprocess(dataset = dataset, subset = subset)
    else:
        preprocess(dataset = dataset, subset = subset)

   

    