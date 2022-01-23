"""Defines a function that computes the metrics for each slice of 
a categorical feature using an input csv containing the test data
and the corresponding predictions. Additionally, this allows for a 
script that can be run on the terminal and saves the result on text.

BY: MIKE DOROSAN, DATE: 2022
"""

import pandas as pd
import numpy as np
import os
from joblib import dump, load
from model import compute_model_metrics

# read test/validation results
test_results = pd.read_csv('data/validation_results.csv')
lb = load('starter/model/lbinarizer.joblib')

def slice_validation(results_df, cat_feature):
    """Function for calculating precision, recall, and fbeta on slices of a 
    cat_feature
    
    Parameters
    ----------
    results_df : pd.DataFrame
        complete validation/test data with a column corresponding to
        model predictions, 'predictions'
    cat_feature : str
        categorical feature to do slicing on
    Returns
    ----------
    per_slice_performance : pd.DataFrame
        per slice performance for the input cat feature arranged in
        a dataframe
    """
    per_slice_performance = pd.DataFrame()
    precisions = []
    recalls = []
    fbetas = []
    classes = []
    for cls in results_df[cat_feature].unique():
        temp = results_df[results_df[cat_feature] == cls]
        y_true = lb.transform(temp['salary'].values)
        y_preds = lb.transform(temp['predictions'].values)

        precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(fbeta)
        classes.append(cls)
    per_slice_performance['classes'] = classes
    per_slice_performance['precision'] = precisions
    per_slice_performance['recall'] = recalls
    per_slice_performance['fbeta'] = fbeta
    
    return per_slice_performance

if __name__ == "__main__":

    cat_feats = [i for i in test_results.columns[test_results.dtypes == object].to_list()]
    cat_feats.remove('salary')
    cat_feats.remove('predictions')

    for feat in cat_feats:
        performance = slice_validation(test_results, feat)
        # create save dir if not exists
        if not os.path.exists('slice_output'):
            os.makedirs('slice_output')
        np.savetxt(
            f'slice_output/{feat}_slice_output.txt', 
            performance.values, 
            fmt=("%s", "%5.5f", "%5.5f", "%5.5f"),
            delimiter='\t',
            header="\t".join(performance.columns.to_list())
        )