"""This script allows for testing the API POST (inference) on terminal
using hard coded data shown below:
"""
import requests
import json

data_low = {
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

r = requests.post(
    "http://127.0.0.1:8000/inferences/", 
    data=json.dumps(data_low)
)
print(data_low)
print(r.json())

data_high = {
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

r = requests.post(
    "http://127.0.0.1:8000/inferences/", 
    data=json.dumps(data_high)
)
print(data_high)
print(r.json())