from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)
    
    
class Input(BaseModel):
    TV: float

@app.post("/predict")
def tv_sales_prediction(input: Input):
    prediction = model.predict([[input.TV]])
    return {"prediction": prediction}

# Run the API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
