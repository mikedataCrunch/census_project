# Welcome
This repository contains a demonstration of basic CI/CD principles in three complementary components explained below--after some introduction on the set-up and dataset.

* GET greeting: https://census-mdorosan2022.herokuapp.com/
* Docs to try out the POST inference: https://census-mdorosan2022.herokuapp.com/docs

## Set-up
### Repository
- Clone this repository

### Environment
- Download and install conda if you don’t have it yet
- Use the supplied requirements file to create a new environment, or
- Run the following in your Linux terminal

```bash
conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter\
 jupyterlab fastapi uvicorn -c conda-forge
```
- Install git either through conda (`conda install git`) or through your CLI, e.g., `sudo apt-get git` .
- Install heroku cli: https://devcenter.heroku.com/articles/heroku-cli
- Initialize `dvc` using `dvc init`

### AWS S3
- In your CLI environment install the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html).
- In the navigation bar in the Udacity classroom select **Open AWS Gateway** and then click **Open AWS Console**. You will not need the AWS Access Key ID or Secret Access Key provided here.
- From the Services drop down select S3 and then click Create bucket.
- Give your bucket a name, the rest of the options can remain at their default.
- Create a folder in that bucket and get the S3 URI of that folder e.g., `s3://mdorosan2022/census_classification/` for use in config files like that of DVCs.

To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found [here](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console), what follows is a paraphrasing:

- Sign in to the IAM console [here](https://console.aws.amazon.com/iam/) or from the Services drop down on the upper navigation bar.
- In the left navigation bar select **Users**, then choose **Add user**.
- Give the user a name and select **Programmatic access**.
- In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
- Tags are optional and can be skipped.
- After reviewing your choices, click create user.
- Configure your AWS CLI to use the Access key ID and Secret Access key, follow the prompts after the command: `aws configure`


### **GitHub Actions**
- Setup GitHub Actions on your repository. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
- Make sure you set up the GitHub Action to have the same version of Python as you used in development. The action `.yml` file can be modified accordingly.
- Add your [AWS credentials to the Action](https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions).
- Set up [DVC in the action](https://github.com/iterative/setup-dvc) and specify a command to `dvc pull`.
- These are implemented in the `.github/worfklows/python-app.yml` file.

### DVC: Local use
- Do a `dvc pull`

### API Deployment: Heroku
- Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
- Ensure that heroku cli recognizes your account, CLI run: `heroku login`
- Create a new app and have it deployed from your GitHub repository.
    - Enable automatic deployments that only deploy if your continuous integration passes.
    - Hint: think about how paths will differ in your local environment vs. on Heroku.
    - Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
- Set up DVC on Heroku using the instructions contained in the starter directory.
- Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
- Write a script that uses the requests module to do one POST on your live API.

## Dataset
Data description can be found here: https://archive.ics.uci.edu/ml/datasets/census+income

## Component 1: CLI Model Training and Evaluation
A Logistic Regression model implemented in sci-kit learn is trained and evaluated using the following python scripts:

* `train_model.py` 
    1. reads the raw data from `data/census_clean.csv` 
    2. splits the data into an 80-20 train-test split 
    3. performs one-hot encoding and label binarization on the categorical and target columns
    4. trains the model
    5. performs inference on the test set
    6. saves the following: 
        - `data/validation_results.csv` - inference results
        - `starter/model/lbinarizer.joblib` - fitted binarizer for the target column
        - `starter/model/encoder.joblib` - fitted OHE for the categorical columns
        - `starter/model/model.joblib` - fitted model for inference use

* `starter/evaluate.py`
    1. reads the `data/validation_results.csv`
    2. calculates the metrics using the `compute_model_metrics` from the `starter.ml.model` package per slice of each categorical column.
    3. saves the results as comma-separated values in text files per categorical column in the `data/slice_output` directory.

### DVC Pipeline
Alternatively, you can run the following command to do the above through `dvc`. This creates the necessary `dvc` files that tracks all the dependencies and outputs resulting from the run.

```
dvc run -n train --force -d starter/train_model.py -d data/census_clean.csv -o starter/model/encoder.joblib -o starter/model/lbinarizer.joblib -o starter/model/model.joblib -o data/validation_results.csv python starter/train_model.py
```
At the end of the run, all the tracked files will be uploaded to the remote using: `dvc push`

## Component 2: FastAPI App
This component demonstrates the launching of a fast API app that (1) does a welcoming message on a GET request in the root path `/` and (2) does inference on a POST request on the `/inferences/` path.

To run ensure that model dependencies are in place and run this command in the CLI: `uvicorn main:app --reload`

The `--reload` makes the app reload whenever changes are detected in the directory files.

To check, you can run a POST using the `requests` to upload data for inference. You may use the provided script for this contained in `starter/tests/live_post.py`. CLI run: `python starter/tests/live_post.py`. Expect a returned prediction on the uploaded data shown in the CLI.

You can visit the app here, for the welcome greeting (GET): http://127.0.0.1:8000
View the docs here, and try out the execution of the inference: http://127.0.0.1:8000/docs

## Component 3: Heroku App
This link demonstrates the deployment of the Heroku App in the CLI: https://devcenter.heroku.com/articles/git

Issues encountered: Python buildpack not installed, can be fixed by explictly putting the the python app buildpack as a heroku CLI command or by setting it up in the dashboard settings of Heroku.

### Other notes
- Redeployment of heroku app with no changes in repo.
```
git commit --allow-empty -m "Trigger Heroku deploy after enabling collectstatic"
git push heroku master
```

## End
