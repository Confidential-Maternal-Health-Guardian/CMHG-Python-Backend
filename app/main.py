from dp_ml import RandomForest, DPRandomForest

from fastapi import FastAPI
import uvicorn

app = FastAPI()



@app.get("/")
def hello_world():
    return "Hello World"


if __name__ == '__main__':
    dpr = DPRandomForest()
    rf = RandomForest()

    print('initialized dpr, rf')
    dpr.train()
    rf.train()
    print('trained dpr, rf')
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)