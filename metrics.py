import load
import some_utils
import numpy as np

import config_parser as cp

config_values = cp.parse()
    
def get_classification(df, dfpred, loc_emotions, k=1):
   
    binary_true=(df[loc_emotions] >= 0.5).astype(int)
    threshold = np.sort(dfpred[loc_emotions].values, axis = 1)[:,-k]
    binary_pred = dfpred[loc_emotions].ge(threshold, axis='rows').astype(int)
    if k < len(loc_emotions):
        binary_pred[binary_pred.sum(axis = 1) == len(loc_emotions)] = 0

    binary_pred = binary_pred.loc[(binary_true[loc_emotions]!=0).any(axis=1)]
    binary_true = binary_true.loc[(binary_true[loc_emotions]!=0).any(axis=1)]

    return binary_true, binary_pred

    
def get_classification_quality(binary_true, binary_pred, loc_emotions):
   
    intersection = (binary_true[loc_emotions] * binary_pred[loc_emotions]).sum(axis = 1)
    union = ((binary_true[loc_emotions] + binary_pred[loc_emotions]) > 0).astype(int).sum(axis = 1)
    jaccard = intersection/union
    labelset_P = intersection/binary_pred[loc_emotions].sum(axis = 1).fillna(0)
    labelset_R = intersection/binary_true[loc_emotions].sum(axis = 1).fillna(0)
    labelset_F = (2*labelset_P*labelset_R/(labelset_P + labelset_R)).fillna(0)
    return np.mean(jaccard), np.mean(labelset_F), np.mean(labelset_P), np.mean(labelset_R)


if __name__ == "__main__":
   pass