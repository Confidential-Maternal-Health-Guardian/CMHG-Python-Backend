from dp_ml import RandomForest, DPRandomForest, SGD, DPSGD

from fastapi import FastAPI
import uvicorn
import warnings

warnings.filterwarnings("ignore")

app = FastAPI()



@app.get("/")
def hello_world():
    return "Hello World"


if __name__ == '__main__':
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

    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)