import json
import numpy as np
import os
from sklearn.externals import joblib
def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_hearth.pkl')
    print("Found model:", os.path.isfile(model_path)) #To check whether the model is actually present on the location we are looking at
    model = joblib.load(model_path)
def run(data):
    try:
        data = np.array(json.loads(data))
        result = model.predict(data)
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
