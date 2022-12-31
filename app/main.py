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

class Info(BaseModel):
    Age: int
    SystolicBP: int
    DiastolicBP: int
    BS: float
    BodyTemp: float
    HeartRate: int
    model_type: str

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
    if info_dict['model_type'] == "sgd":
        classifier = sgd
    elif info_dict['model_type'] == "dpsgd":
        classifier = dpsgd
    elif info_dict['model_type'] == "rf":
        classifier = rf
    elif info_dict['model_type'] == "dpr":
        classifier = dpr
    else:
        raise Exception

    del info_dict['model_type']

    df = pd.DataFrame(info_dict, index=[0])
    
    predict_n = classifier.predict(df)

    response = Res(prediction=predict_n)
    return response


if __name__ == '__main__':


    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)