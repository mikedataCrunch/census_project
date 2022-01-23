import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

from sklearn.base import is_classifier

@pytest.fixture
def raw():
    raw_data = pd.read_csv('data/census.csv')
    return raw_data

def test_raw_type(raw):
    """Raw data should be readable into pd.DataFrame()"""
    assert type(raw)==type(pd.DataFrame())


@pytest.fixture
def cleaned():
    cleaned_data = pd.read_csv('data/census_clean.csv')
    return cleaned_data

def test_cleaned_type(cleaned):
    """Processed data should be readable into pd.DataFrame()"""
    assert type(cleaned)==type(pd.DataFrame())

def test_cleaned_na(cleaned):
    """Test if processed contains to nulls"""
    assert cleaned.shape == cleaned.dropna().shape, "Dropping null changes shape."


@pytest.fixture
def sample_data():
    sample_df = pd.read_csv('data/census_clean.csv', nrows=100)
    return sample_df

@pytest.fixture
def processed(sample_data):
    cat_cols = [i for i in sample_data.columns[sample_data.dtypes == object].to_list() \
         if i != 'salary']
    processed_data = process_data(
        sample_data, 
        categorical_features=cat_cols, 
        label='salary',
        training=True
    )
    return processed_data

def test_processed_defaults(processed):
    """Test processing function output"""
    assert type(processed)==tuple, "Not a tuple"
    assert type(processed[0]) == type(np.array([])), "Supposed input data not a np.array"
    assert type(processed[1]) == type(np.array([])), "Supposed input data not a np.array"
    assert type(processed[2]) == type(OneHotEncoder()), "Supposed one-hot encoder not of expected type"
    assert type(processed[3]) == type(LabelBinarizer()), "Supposed binarizer not of expected type"


    

def test_model_train(processed):
    """Test model training output to be a classifier"""
    X, y, _, _ = processed
    model = train_model(X, y)
    assert is_classifier(model), "Supposed classifier model is not a classifier"

@pytest.fixture
def sample_results():
    y = np.random.choice([0, 1], size=10, replace=True)
    preds = np.random.choice([0, 1], size=10, replace=True)
    return y, preds

def test_model_eval(sample_results):
    """Test metric evaluation functions"""
    y, preds = sample_results
    metrics = compute_model_metrics(y, preds)
    assert len(metrics)==3, "Returned more than the expected number of metrics"