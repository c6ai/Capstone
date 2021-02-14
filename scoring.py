import joblib, json, logging, os, pickle
import numpy as np
import pandas as pd
import azureml.automl.core
from azureml.automl.core.shared import logging_utilities, log_server
from azureml.telemetry import INSTRUMENTATION_KEY
from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

input_sample = pd.DataFrame({"Column1": pd.Series([0], dtype="int64"), "Column2": pd.Series([0], dtype="int64"), "Column3": pd.Series([0], dtype="int64"), "Column4": pd.Series([0], dtype="int64"), "Column5": pd.Series([0], dtype="int64"), "Column6": pd.Series([0], dtype="int64"), "Column7": pd.Series([0], dtype="int64"), "Column8": pd.Series([0.0], dtype="float64"), "Column9": pd.Series([0], dtype="int64"), "Column10": pd.Series([0], dtype="int64"), "Column11": pd.Series([0], dtype="int64"), "Column12": pd.Series([0], dtype="int64"), "Column13": pd.Series([0.0], dtype="float64"), "Column14": pd.Series([0], dtype="int64"), "Column15": pd.Series([0.0], dtype="float64"), "Column16": pd.Series([0], dtype="int64"), "Column17": pd.Series([0], dtype="int64"), "Column18": pd.Series([0], dtype="int64"), "Column19": pd.Series([0], dtype="int64"), "Column20": pd.Series([0], dtype="int64"), "Column21": pd.Series([0], dtype="int64"), "Column22": pd.Series([0], dtype="int64"), "Column23": pd.Series([0], dtype="int64"), "Column24": pd.Series([0], dtype="int64"), "Column25": pd.Series([0.0], dtype="float64"), "Column26": pd.Series([0.0], dtype="float64"), "Column27": pd.Series([0.0], dtype="float64"), "Column28": pd.Series([0.0], dtype="float64"), "Column29": pd.Series([0], dtype="int64"), "Column30": pd.Series([0.0], dtype="float64"), "Column31": pd.Series([0.0], dtype="float64"), "Column32": pd.Series([0], dtype="int64"), "Column33": pd.Series([0], dtype="int64"), "Column34": pd.Series([0.0], dtype="float64"), "Column35": pd.Series([0], dtype="int64"), "Column36": pd.Series([0], dtype="int64"), "Column37": pd.Series([0.0], dtype="float64"), "Column38": pd.Series([0], dtype="int64")})
output_sample = np.array([0])
try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except:
    pass

def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the model file back
    # into a sklearn model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({'model_name': path_split[-3], 'model_version': path_split[-2]})
    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise

@input_schema('data', PandasParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
