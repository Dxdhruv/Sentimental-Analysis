from preprocessing import basic_text_cleaning
from config import FULL_MODEL_PATH, EMOTION_LABELS
import dill
import torch
print("torch version:", torch.__version__)
from fastapi import FastAPI
from pydantic import BaseModel
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import sys
import importlib

app = FastAPI()

class TextInput(BaseModel):
    text: str
    
@app.post("/pred")
def predict_emotion(input_data: TextInput):
    try:
        print("Torch Version:", torch.__version__)
        model = torch.load(FULL_MODEL_PATH, map_location=torch.device('cpu'), pickle_module=dill)
        model.eval()
        text = basic_text_cleaning(input_data.text)
        answer = model.predict(text)
        
        if isinstance(answer[0], tuple):
            value = answer[0][0]
        else:
            value = answer[0]
            
        if isinstance(value, torch.Tensor):
            prediction_label = EMOTION_LABELS[value.item()]
        elif isinstance(value, int):
            prediction_label = EMOTION_LABELS[value]
        elif isinstance(value, str):
            prediction_label = value
        else:
            raise ValueError("Unexpected type for prediction label")
        
        return {"emotion": prediction_label}
    except Exception as e:
        print(f"error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)