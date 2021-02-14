import joblib
import pandas as pd
from azureml.core.model import Model
def init():
    global automl_model
    automl_model_path = Model.get_model_path('automl_model')
    automl_model = joblib.load(automl_model_path)
def run(raw_data):
    data = pd.read_json(raw_data, orient='records')
    predictions = automl_model.predict(data)
    foresights = predictions
    return {'predictions': predictions.tolist()}
