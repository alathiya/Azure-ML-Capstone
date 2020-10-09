import json
import numpy as np
import os
import pickle
import joblib
import pandas as pd

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best-trained-model.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        #data = np.array(json.loads(data))
        #result = model.predict(data)
        ## You can return any data type, as long as it is JSON serializable.
        #return result.tolist()

        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = model.predict(data)
        return json.dumps({"result": result.tolist()})

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
