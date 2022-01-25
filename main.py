from fastapi import Body, FastAPI

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Instantiate the app.
app = FastAPI()

# Declare the data object with its components and their type.
class Input(BaseModel):
    inference_id : int
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
    hours_per_week : str
    native_country : str

# Define a GET on the specified endpoint.
@app.get("/")
async def greet_root():
    return {"greeting": "Hello World!"}

@app.post("/inferences/")
async def create_input(
    input : Input = Body(
        ...,
        example={
            "inference_id": 0,
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
    return input

@app.get("/inferences/{inference_id}")
async def get_salary(
    inference_id : int
):
    # insert code of predicting salary here
    return {
        "salary": f"Predicted salary for {inference_id} is ..."}



# import os

# if "DYNO" in os.environ and os.path.isdir(".dvc"):
#     os.system("dvc config core.no_scm true")
#     if os.system("dvc pull") != 0:
#         exit("dvc pull failed")
#     os.system("rm -r .dvc .apt/usr/lib/dvc")

