import joblib
import json
import numpy as np
import os

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_hearth.pkl')
    model = joblib.load(model_path)

input_sample = np.array([[75, 0, 582, 0, 20, 1, 265000, 1.9, 130, 1, 0, 4, 1]])
output_sample = np.array([0])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))

def run(data):
    try:
        result = model.predict(data)

        return "Here is your result = " + str(result)
    except Exception as e:
        error = str(e)
        return error
