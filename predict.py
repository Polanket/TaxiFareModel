import os
import joblib
import pandas as pd
from math import sqrt
# GCloud imports
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error
from TaxiFareModel.params import STORAGE_LOCATION, BUCKET_NAME

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"


def get_test_data(nrows, data="s3"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df


def get_model(gcloud=True):
    if gcloud:
        client = storage.Client()
        bucket = client.get_bucket(BUCKET_NAME)
        cloud_model = bucket.get_blob(STORAGE_LOCATION)
        cloud_model.download_to_filename('cloud_model.joblib')
        model_path = 'cloud_model.joblib'
    else:
        model_path = PATH_TO_LOCAL_MODEL
    pipeline = joblib.load(model_path)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(nrows)
    pipeline = get_model(gcloud=True)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':
    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    nrows = 10000
    generate_submission_csv(nrows, kaggle_upload=False)
