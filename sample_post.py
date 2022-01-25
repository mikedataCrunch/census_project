import requests
import json

data = {
    "inference_id": 0,
    # insert expected feature inputs here
    "age" : 50,
    "workclass" : "Self-emp-not-inc",
    "fnlgt" : 83311,
    "education" : "Bachelors",
    "education_num" : 13,
    "marital_status" : "Married-civ-spouse",
    "occupation" : "Exec-managerial",
    "relationship" : "Husband",
    "race" : "White",
    "sex" : "Male",
    "capital_gain" : 0,
    "capital_loss" : 0,
    "hours_per_week" : 13,
    "native_country" : "United-States" 
}


r = requests.post("http://127.0.0.1:8000/inferences/", data=json.dumps(data))

print(r.json())

# use this to allow for inputs in terminal
# if __name__ == "__main__":
