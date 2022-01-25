from fastapi import Body, FastAPI

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd
from joblib import load
# # dvc set up with heroku
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Instantiate the app.
app = FastAPI()

# Declare the data object with its components and their type.
class Item(BaseModel):
    # insert type hinting here for feature inputs
    age: int
    workclass: str 
    fnlgt: int
    education : str
    education_num : int
    marital_status : str
    occupation : str
    relationship : str
    race : str
    sex : str
    capital_gain : Union[int, float]
    capital_loss : Union[int, float]
    hours_per_week : int
    native_country : str

# Define a GET on the specified endpoint.
@app.get("/")
async def greet_root():
    return {"greeting": "Hello World!"}

@app.post("/inferences/")
async def predict(
    item : Item = Body(
        ...,
        example={
            # insert expected feature inputs here
            "age" : 39,
            "workclass" : "State-gov",
            "fnlgt" : 77516,
            "education" : "Bachelors",
            "education_num" : 13,
            "marital_status" : "Never-married",
            "occupation" : "Adm-clerical",
            "relationship" : "Not-in-family",
            "race" : "White",
            "sex" : "Male",
            "capital_gain" : 2174,
            "capital_loss" : 0,
            "hours_per_week" : 40,
            "native_country" : "United-States" 
        },
    ),
):  
    item = {k : [v] for k, v in item.dict().items()}
    test = pd.DataFrame.from_dict(item)
    # load inputs
    encoder = load('starter/model/encoder.joblib')
    lb = load('starter/model/lbinarizer.joblib')

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

    # proces the test data with the process_data function.
    test, _, _, _ = process_data(
        test, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )

    # load the model
    model = load('starter/model/model.joblib')

    salary = inference(model=model, X=test)
    salary = lb.inverse_transform(salary)[0] # index item to get str out

    return {"salary": salary}
