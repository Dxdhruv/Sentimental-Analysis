from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Define the TextInput schema here directly
class TextInput(BaseModel):
    text: str

# Import your model and preprocessing functions
from model_loader import ModelPredictor, load_model
from preprocessing import basic_text_cleaning

app = FastAPI()

# Configure CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/action")
def predict_emotion(input_data: TextInput):
    text = basic_text_cleaning(input_data.text)
    try:
        model = load_model()
        predictor = ModelPredictor(model=model)
        predicted_label = predictor.predict(text)
        
        # Return sentiment in a consistent format that matches frontend expectations
        return {"sentiment": predicted_label.lower()}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)