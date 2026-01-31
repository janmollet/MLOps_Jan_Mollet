from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data


app = FastAPI()

class WineData(BaseModel):
    proline: float
    flavanoids: float
    color_intensity: float
    

class WineResponse(BaseModel):
    response:int

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
    wine_features.flavanoids,     
    wine_features.color_intensity, 
    wine_features.proline         
]]

        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    
