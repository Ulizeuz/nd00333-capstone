import joblib
import json
import numpy as np
import os

from inference_schema.schema_decorators import input_schema, output_schema
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment. Join this path with the filename of the model file.
    # It holds the path to the directory that contains the deployed model (./azureml-models/$MODEL_NAME/$VERSION)
    # If there are multiple models, this value is the path to the directory containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_hearth.pkl')
    # Deserialize the model file back into a model.
    model = joblib.load(model_path)

    global name
    # Note here, the entire source directory from inference config gets added into image.
    # Below is an example of how you can use any extra files in image.
    with open('./source_directory/extradata.json') as json_file:
        data = json.load(json_file)
        name = data["people"][0]["name"]

input_sample = np.array([[75, 0, 582, 0, 20, 1, 265000, 1.9, 130, 1, 0, 4, 1]])
output_sample = np.array([0])

@input_schema('data', NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data):
    try:
        result = model.predict(data)
        # You can return any JSON-serializable object.
        return "Here is your result = " + str(result)
    except Exception as e:
        error = str(e)
        return error
