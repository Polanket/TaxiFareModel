import joblib
from termcolor import colored
from memoized_property import memoized_property
# ML flow imports
import mlflow
from mlflow.tracking import MlflowClient
# Sklearn imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Internal imports
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

MLFLOW_URI = "https://mlflow.lewagon.co/"
myname = "polanco"
EXPERIMENT_NAME = f"TaxifareModel_{myname}"

class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # For ML flow
        self.experiment_name = EXPERIMENT_NAME

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def set_experiment_name(self, experiment_name):
        """Defines the name of the experiment for ML Flow"""
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_trans', TimeFeaturesEncoder('pickup_datetime')),
            ('time_ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        pre_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude",
                                     "pickup_longitude",
                                     'dropoff_latitude',
                                     'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder='drop')

        self.pipeline = Pipeline([
            ('processing', pre_pipe),
            ('linear_model', RandomForestRegressor())
        ])

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    def save_model(self):
        """Save the model as .joblib file"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # get data
    data = get_data(50_000)
    # clean data
    clean_data = clean_data(data)
    # set X and y
    y = clean_data['fare_amount']
    X = clean_data.drop('fare_amount', axis=1)
    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # train
    trainer = Trainer(X_train, y_train)
    trainer.set_experiment_name('Rndm Forest')
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    # evaluate
    print(rmse)
    trainer.save_model()