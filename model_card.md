# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model contained in this project is a Logistic Regression model implemented through sci-kit learn using default parameters. 

## Intended Use
The model is intended to predict a U.S. citizen's salary given the following information:
* `age`: 37,
* `workclass`: "State-gov",
* `fnlgt`: 77516,
* `education`: "Bachelors",
* `education_num`: 13,
* `marital_status`: "Never-married",
* `occupation`: "Adm-clerical",
* `relationship`: "Not-in-family",
* `race`: "White",
* `sex`: "Male",
* `capital_gain`: 2174,
* `capital_loss`: 0,
* `hours_per_week`: 40,
* `native_country`: "United-States"


## Training Data
The model was trained using a 80% randomly selected without replacement subset of the original data.

## Evaluation Data
The model's performance was evaluated using the remaining 20%.  These metrics were calculated according to slices of each categorical feature (e.g., educational attainment, gender)--reported in the directory `data/slice_output/` as `.txt` files of comma-seperated values.

## Metrics
Here are the model performance on the following metrics (high: 1.0, low: 0.0): 
|-|-|-|
|precision|recall|fbeta|
|0.73|0.26|0.37|

## Ethical Considerations

## Caveats and Recommendations