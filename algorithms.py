import os
import load
import some_utils as ut
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import eigs

import config_parser as cp
config_values = cp.parse(os.path.dirname(os.path.realpath(__file__)))


def ester(S, C, M, d=0.85, mode='sent2emo', lexnorm=False):
    """
    Implements ESTeR algorithm

    Parameters
    ----------
    S : sparse dataset matrix (sentences x vocabulary)
    C : sparse lexicon matrix (vocabulary x emotions)
    M : sparse symmetric co-occurrence matrix (vocabulary x vocabulary)
    d : probability not to restart in the random walk model
    mode : random walk version
        'sent2emo': from a sentence to a lexicon
        'emo2sent': from a lexicon to a sentence
    lexnorm : normalize lexicon by l1 or not

    Returns
    -------
    dense matrix of scores (sentences x emotions)
    """

    if mode == 'sent2emo':
        M = (identity(M.shape[0], dtype='float') - d*ut.row_normalize(M, norm="l1")).tocsc()

    elif mode == 'emo2sent':
        M = (identity(M.shape[0], dtype='float') - d*ut.column_normalize(M, norm="l1")).tocsc()

    if lexnorm:
        C = ut.column_normalize(C, norm="l1")
    
    X = spsolve(M, C)

    S = ut.row_normalize(S, norm="l1")
    return (S.dot(X)).todense()*(1-d)


def cos_lexicon(S, C):
    """
    Implements cosine similarity with a lexicon

    Parameters
    ----------
    S : sparse dataset matrix (sentences x vocabulary)
    C : sparse lexicon matrix (vocabulary x emotions)
    
    Returns
    -------
    dense matrix of scores (sentences x emotions)
    """

    S = ut.row_normalize(S, norm='l2')
    C = ut.column_normalize(C, norm='l2')
    return (S.dot(C)).todense()

def cos_embedding(S, C, E):
    """
    Implements cosine similarity with a lexicon in the latent space

    Parameters
    ----------
    S : sparse dataset matrix (sentences x vocabulary)
    C : sparse lexicon matrix (vocabulary x emotions)
    E : sparse embedding matrix (vocabulary x vocabulary)
    
    Returns
    -------
    dense matrix of scores (sentences x emotions)]
    """

    S = ut.row_normalize(S.dot(E), norm='l2')
    C = ut.column_normalize(E.transpose().dot(C), norm='l2')
    return (S.dot(C)).todense()


if __name__ == "__main__":
   pass