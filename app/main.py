import sys
import os
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import warnings
import pandas as pd
import time
import torch
sys.path.append(os.getcwd())
from app.dp_ml import RandomForest, DPRandomForest, SGD, DPSGD

warnings.filterwarnings("ignore")


class Req(BaseModel):
    age: int
    systolicBp: int
    diastolicBp: int
    bs: float
    bodyTemp: float
    heartRate: int
    modelType: str
    epsilon: float


class Res(BaseModel):
    riskLevel: str


app = FastAPI()

models = {}
riskLevels = {0:"low risk", 1:"mid risk", 2:"high risk"}

@app.on_event('startup')
def init_models():
    models['dpr'] = DPRandomForest()
    models['rf'] = RandomForest()
    models['dpsgd'] = {}
    models['sgd'] = SGD()
    print('models initialized')
    epsilons = [0.5, 1.0, 1.5, 2.0]

    for epsilon in epsilons:
        dpsgd = DPSGD(epsilon=epsilon, delta=1e-8)
        model_path = os.path.join(os.getcwd(), "models", "dpsgd_" + str(epsilon).replace('.','_') + ".pt")
        dpsgd.model.load_state_dict(torch.load(model_path))
        models['dpsgd'][str(epsilon)] = dpsgd

    model_path = os.path.join(os.getcwd(), "models", "sgd.pt")
    models['sgd'].model.load_state_dict(torch.load(model_path))
    
    models['dpr'].train()
    models['rf'].train()
    print('models trained')
    return models


@app.get("/")
def hello_world():
    return "Hello World"

@app.post("/predict")
async def predict(body: Req):
    classifier = None
    
    model_selection = body.modelType
    epsilon = body.epsilon

    if model_selection in models:
        classifier = models[model_selection]
        if type(classifier) is dict:
            classifier = classifier[str(epsilon)]
    else:
        raise Exception

    info = body.dict()
    del info['modelType']
    del info['epsilon']
    
    df = pd.DataFrame(info, index=[0])
    
    prediction = classifier.predict(df)


    response = Res(riskLevel=riskLevels[prediction])
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5001, reload=True)