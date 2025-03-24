import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import requests

# our scripts
from helpers import dill_load

KEYS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
CLASSES = ["setosa", "versicolor", "virginica"]
MODEL_PATH = "./"
MODEL_NAME = "iris_model"

app = FastAPI()


def load_model():
    return dill_load(MODEL_PATH, MODEL_NAME)


@app.get("/")
def status():
    return {"status": "ok"}


@app.post("/predict")
async def predict(features: dict):
    """
    Example:
        features = {"sepal length (cm)" : 4.5,
                    "sepal width (cm)"  : 2.5,
                    "petal length (cm)" : 4.0,
                    "petal width (cm)"  : 1.5}
    """

    try:
        # generate feature array
        f_array = np.array([[features[key]] for key in KEYS]).T

        # Make the prediction
        model = load_model()
        prediction = int(model.predict(f_array))  # integer (0, 1, 2)

        # Get the predicted class
        predicted_class = CLASSES[prediction]

        return JSONResponse(
            content={"features": features, "prediction": predicted_class}
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=400, detail=f"Cannot run model for the given features!"
        )
