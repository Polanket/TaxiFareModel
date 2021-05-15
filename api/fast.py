from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from predict import get_model
import pandas as pd
import csv

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_credentials=True,
                   allow_methods=['*'],
                   allow_headers=['*']
                   )

@app.get('/')
def index():
    return {'message': 'Welcome to ML TaxiFare API'}


@app.post('/predict_fare')
async def predict_fare(features: dict = Body(...)):
    features = parse_body(features)
    print(features)
    print(features.dtypes)
    model = get_model(gcloud=True)
    prediction = model.predict(features)
    return {"prediction": prediction[0]}


def parse_body(body):
    feature_dict = {
        "key": [body['key']],
        "pickup_datetime": [body['pickup_datetime']],
        "pickup_longitude": [float(body['pickup_longitude'])],
        "pickup_latitude": [float(body['pickup_latitude'])],
        "dropoff_longitude": [float(body['dropoff_longitude'])],
        "dropoff_latitude": [float(body['dropoff_latitude'])],
        "passenger_count": [int(body['passenger_count'])]
    }
    df = pd.DataFrame.from_dict(feature_dict)
    return df
