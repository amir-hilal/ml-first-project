from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc

app = FastAPI()

mlflow.set_tracking_uri("http://localhost:5000")

model_name = "IrisRandomForestModel"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/latest")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    try:
        input_features = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]]
        prediction = model.predict(input_features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API with: uvicorn endpoint:app --reload
