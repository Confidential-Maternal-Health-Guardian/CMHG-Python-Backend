from dp_ml import RandomForest, DPRandomForest, SGD, DPSGD

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

app = FastAPI()

dpr = None
rf = None
sgd = None
dpsgd = None

dpr = DPRandomForest()
rf = RandomForest()
dpsgd = DPSGD(epsilon=1, delta=1e-8)
sgd = SGD()
print('models initialized')

dpr.train()
rf.train()

dpsgd_accuracy = dpsgd.train(synthetic=False)
print("DPSGD Accuracy: ", dpsgd_accuracy)

sgd_accuracy = sgd.train(synthetic=False)
print("SGD Accuracy: ", sgd_accuracy)
print('models trained')

class Res(BaseModel):
    prediction: int

@app.get("/")
def hello_world():
    return "Hello World"

@app.post("/predict/")
async def predict(info: Request):
    classifier = None
    info_dict = await info.json()
    info_dict = info_dict['query']
    print("Reguest:", info_dict)

    model_selection = info_dict['ModelType']
    epsilon = info_dict['epsilon']

    if model_selection == "sgd":
        classifier = sgd
    elif model_selection == "dpsgd":
        classifier = dpsgd
    elif model_selection == "rf":
        classifier = rf
    elif model_selection == "dpr":
        classifier = dpr
    else:
        raise Exception

    del info_dict['ModelType']
    del info_dict['epsilon']

    df = pd.DataFrame(info_dict, index=[0])
    
    predict_n = classifier.predict(df)

    response = Res(prediction=predict_n)
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)