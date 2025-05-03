import torch
import dill
from transformers import AutoTokenizer, ModernBertModel
from config import MODEL_PATH, MODEL_NAME, EMOTION_LABELS, device
from preprocessing import basic_text_cleaning  # Uncomment if you want text preprocessing

class ModernBertClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(ModernBertClassifier, self).__init__()
        self.bert = ModernBertModel.from_pretrained(MODEL_NAME)
        
        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Classification layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(256),
            torch.nn.Dropout(0.3),
            
            torch.nn.Linear(256, num_labels)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# Load model function
def load_model():
    print("Loading model...")
    model = torch.load(MODEL_PATH, map_location=device, pickle_module=dill)
    model.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded successfully.")
    return model

# Load model once to optimize performance
model = load_model()

# Prediction function
def model_predict(text):
    # cleaned_text = basic_text_cleaning(text)  # Uncomment if needed
    print(f"Processing Text: {text[:50]}....")
    try:
        encoding = model.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**encoding).cpu()
            prediction = torch.argmax(logits, dim=-1)

        predicted_label = EMOTION_LABELS[prediction[0].item()]
        return predicted_label
    except Exception as e:
        print(f"Error in model_predict: {str(e)}")
        raise
