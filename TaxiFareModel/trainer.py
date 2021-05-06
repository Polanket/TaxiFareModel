from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

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
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        return compute_rmse(y_pred, y_test)


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
    trainer.run()
    rmse = trainer.evaluate(X_test, y_test)
    # evaluate
    print(rmse)