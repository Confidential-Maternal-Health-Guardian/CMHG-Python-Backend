from dp_ml import RandomForest, DPRandomForest, SGD, DPSGD

from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import warnings
import pandas as pd
import time

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
    confidence: float


app = FastAPI()

models = {}
riskLevels = {0:"low risk", 1:"mid risk", 2:"high risk"}

@app.on_event('startup')
def init_models():
    models['dpr'] = DPRandomForest()
    models['rf'] = RandomForest()
    models['dpsgd'] = DPSGD(epsilon=1, delta=1e-8)
    models['sgd'] = SGD()
    print('models initialized')

    models['dpr'].train()
    models['rf'].train()
    
    start = time.time()
    dpsgd_accuracy = models['dpsgd'].train(synthetic=False)
    end = time.time()
    print(f"DPSGD Accuracy: {dpsgd_accuracy}% {end - start}s")

    start = time.time()
    sgd_accuracy = models['sgd'].train(synthetic=False)
    end = time.time()
    print(f"SGD Accuracy: {sgd_accuracy}% {end - start}s")
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
    else:
        raise Exception

    info = body.dict()
    del info['modelType']
    del info['epsilon']
    
    df = pd.DataFrame(info, index=[0])
    
    prediction = classifier.predict(df)


    response = Res(riskLevel=riskLevels[prediction], confidence=0.5)
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)