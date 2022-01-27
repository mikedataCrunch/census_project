# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from joblib import dump, load
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


# Add code to load in the data.
data = pd.read_csv('data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# save transformers
dump(encoder, 'starter/model/encoder.joblib')
dump(lb, 'starter/model/lbinarizer.joblib')

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
dump(model, 'starter/model/model.joblib')

# load the model
model = load('starter/model/model.joblib')

# predict
preds = inference(model, X_test)
test['predictions'] = lb.inverse_transform(preds)

# test results saved together with original test data with added column
# new col: 'predictions'
test.to_csv('data/validation_results.csv', index=False)
