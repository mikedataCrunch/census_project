from fastapi.testclient import TestClient

# Import our app from main.py.
from main import app
import pytest
import json

# Instantiate the testing client with our app.
client = TestClient(app)

# Write tests using the same syntax as with the requests module.
def test_get_root():
    r = client.get('/')
    assert r.status_code == 200

def test_get_contents():
    r = client.get('/')
    assert r.json() == {"greeting": "Hello World!"}
    

@pytest.fixture
def data_low():
    data = {
    # insert expected feature inputs here
    "age" : 37,
    "workclass" : "Private",
    "fnlgt" : 635913,
    "education" : "Bachelors",
    "education_num" : 13,
    "marital_status" : "Never-married",
    "occupation" : "Exec-managerial",
    "relationship" : "Not-in-family",
    "race" : "Black",
    "sex" : "Male",
    "capital_gain" : 0,
    "capital_loss" : 0,
    "hours_per_week" : 60,
    "native_country" : "United-States" 
}
    return data

def test_inference_low(data_low):
    r = client.post(
        'http://127.0.0.1:8000/inferences/', 
        data=json.dumps(data_low)
    )
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50K"}

@pytest.fixture
def data_high():
    data = {
    # insert expected feature inputs here
    "age" : 36,
    "workclass" : "Private",
    "fnlgt" : 128757,
    "education" : "Bachelors",
    "education_num" : 13,
    "marital_status" : "Married-civ-spouse",
    "occupation" : "Other-service",
    "relationship" : "Husband",
    "race" : "Black",
    "sex" : "Male",
    "capital_gain" : 7298,
    "capital_loss" : 0,
    "hours_per_week" : 36,
    "native_country" : "United-States" 
}
    return data

def test_inference_high(data_high):
    r = client.post(
        'http://127.0.0.1:8000/inferences/', 
        data=json.dumps(data_high)
    )
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}