from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from Batch_Real_inference import predict_batch, predict_real_time2

app = FastAPI()

# 1. Improved Pydantic Model
class InputData(BaseModel):
    utc_timestamp: str
    cet_cest_timestamp: str
    area_offices : float
    area_room_1: float
    area_room_2: float
    area_room_3: float
    area_room_4: float
    compressor: float
    cooling_aggregate: float
    cooling_pumps: float
    dishwasher: float
    ev: float
    grid_import: float
    machine_1: float
    machine_2: float
    machine_3: float
    machine_4: float
    machine_5: float
    pv_facade: float
    pv_roof: float
    refrigerator: float
    ventilation: float


# 2. Batch Prediction Endpoint
@app.post("/predict_batch")
def run_predict_batch(test_batch: List[InputData]):
    try:
        # Convert list of Pydantic models to list of dicts
        batch_data = [item.model_dump() for item in test_batch]

        prediction_df = predict_batch(batch_data)
        # Convert DataFrame to JSON format (List of dicts)
        result_json = prediction_df.to_dict(orient='records')
        return {"prediction_result": result_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 3. Real-Time (Single) Prediction Endpoint
@app.post("/predict_real_time")
def run_predict_real_time(new_data: InputData):
    try:
        # 1. Conversion Pydantic -> Dict
        single_data = new_data.model_dump() if hasattr(new_data, 'model_dump') else new_data.dict()

        prediction_df = predict_real_time2(single_data)

        # Convert DataFrame to JSON format (List of dicts)
        result_json = prediction_df.to_dict(orient='records')

        return {"prediction_result": result_json}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))