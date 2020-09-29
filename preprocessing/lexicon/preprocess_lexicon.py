import sys, os

import pandas as pd
import numpy as np

import argparse

config_path = '../..'
sys.path.append(config_path)

import config_parser as cp
config_values = cp.parse(config_path)

datadir = config_values['LEXICONS_RAW']
outdir = config_values['LEXICONS']
use_emotions = config_values['USE_EMOTIONS']



def write_out(outdir, dataname, df):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    df.to_csv(os.path.join(outdir, dataname +'.txt'), index = False, sep = '\t') 

def preprocess(lexicon):
    if lexicon == 'DepecheMood':
        datafolder = lexicon
        datafile = "DepecheMood_english_token_full"
        df = pd.read_csv(os.path.join(datadir, datafolder, datafile+'.tsv'), sep = '\t', header = 0)
        df.rename(columns={"Unnamed: 0": "word"}, inplace = True)

        mapping = {'fear': ['AFRAID'],\
                    'surprise': ['INSPIRED'],\
                    'sadness': ['SAD'],\
                    'disgust': ['ANNOYED'],\
                    'anger': ['ANGRY'],\
                    'joy': ['AMUSED', 'HAPPY']}

        new_df = pd.DataFrame()
        new_df['word'] = df['word']
        for k, v in mapping.items():
            new_df[k] = df[v].sum(axis = 1)
        new_df.dropna(inplace = True)
        write_out(outdir, lexicon, new_df)
        print('processed {}'.format(lexicon))

    elif lexicon == 'NRC-EmoLex':
        datafolder =  lexicon
        datafile = "NRC-Emotion-Lexicon-Wordlevel-v0.92"
        df = pd.read_csv(os.path.join(datadir, datafolder, datafile + '.txt'), sep = '\t', header=None)
        df = pd.pivot_table(df, columns=[1], values=2, index=[0])
        df.reset_index(inplace=True)
        df.rename(columns={0: "word"}, inplace=True)
        df.drop(columns=['negative', 'positive'], inplace=True)

        write_out(outdir, lexicon, df)
        print('processed {}'.format(lexicon))



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess lexicon into tsv with text fields \"word\" and 8 or 6 emotions.')
    parser.add_argument('-l', '--lexicon', type=str, help='lexicons: DepecheMood, NRC-EmoLex. Use \'all\' to process all datasets.', default='all')
    args = parser.parse_args()
    lexicon = args.lexicon

    if lexicon == 'all':
        preprocess('DepecheMood')
        preprocess('NRC-EmoLex')
    else:
        preprocess(lexicon)
        
  