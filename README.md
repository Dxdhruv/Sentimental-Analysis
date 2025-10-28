# 🧠 Advanced Sentiment Analysis System with ModernBERT

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.34.0-green.svg)](https://fastapi.tiangolo.com)
[![Transformers](https://img.shields.io/badge/Transformers-4.50.3-yellow.svg)](https://huggingface.co/transformers)

A sophisticated **emotion classification system** built using ModernBERT transformer architecture, designed to analyze and categorize text into 5 distinct emotional states: **Happy**, **Sad**, **Aggressive**, **Neutral**, and **Fear**. This project demonstrates advanced NLP techniques, transfer learning, and production-ready API development.

## 🎯 Project Overview

This sentiment analysis system was developed as an **internship project** to address the growing need for automated emotion detection in online content. The system can accurately classify user comments, social media posts, and text content to help identify emotional patterns, detect potentially harmful content, and understand user sentiment at scale.

### Key Features
- **Multi-class Emotion Classification** (5 categories)
- **High Context Length Support** (up to 8192 tokens)
- **Transfer Learning Implementation** with ModernBERT
- **RESTful API** with FastAPI framework
- **Production-ready Architecture** with modular design
- **Comprehensive Testing Suite** with performance metrics

## 🏗️ Technical Architecture

### Model Architecture
```
ModernBERT-base (answerdotai/ModernBERT-base)
├── Frozen BERT Layers (Transfer Learning)
├── Custom Classification Head
│   ├── Linear Layer (768 → 512)
│   ├── ReLU Activation + LayerNorm
│   ├── Dropout (0.3)
│   ├── Linear Layer (512 → 256)
│   ├── ReLU Activation + LayerNorm
│   ├── Dropout (0.3)
│   └── Output Layer (256 → 5 emotions)
└── Tokenizer Integration
```

### System Components
- **`model_loader.py`**: Core model architecture and loading utilities
- **`predictor.py`**: Prediction pipeline and inference logic
- **`preprocessing.py`**: Text cleaning and normalization
- **`api.py`** & **`main.py`**: FastAPI endpoints for model serving
- **`config.py`**: Configuration management and hyperparameters
- **`schemas.py`**: Pydantic models for API validation

## 📊 Dataset & Training

### Custom Dataset Creation
- **Total Samples**: ~6,000 text samples
- **Categories**: 6 emotion classes (Happy, Sad, Angry, Abusive, Neutral, Fear)
- **Sample Distribution**: ~1,000 samples per emotion category
- **Text Length Variation**: 2-3 words to 4,000-5,000 words per sample
- **Data Sources**: Web scraping + synthetic data generation
- **Labeling**: Automated labeling using LLM for consistency

### Training Approach
- **Transfer Learning**: Leveraged pre-trained ModernBERT weights
- **Frozen BERT Layers**: Optimized for limited GPU resources
- **Custom Classification Head**: Fine-tuned for emotion classification
- **CrossEntropy Loss**: Standard multi-class classification loss
- **Batch Processing**: Efficient training with variable-length sequences

## 🚀 Performance Metrics

### Model Performance
- **Overall Accuracy**: ~80%
- **Evaluation Method**: Confusion Matrix + Cross-Validation
- **Inference Speed**: Optimized for real-time predictions
- **Context Length**: Supports up to 8192 tokens
- **Memory Efficiency**: CPU-optimized inference

### Testing Results
The model was extensively tested with diverse text samples including:
- Short phrases (2-3 words)
- Medium-length sentences
- Long-form paragraphs (4000+ words)
- Various emotional contexts and linguistic patterns

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11+
- PyTorch 2.6.0
- CUDA support (optional, for GPU acceleration)

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd Sentimental-Analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download model files**
```bash
# Download the pre-trained model from the provided Google Drive link
# Place model.pth and full_model.pth in the project directory
```

4. **Update configuration**
```python
# Edit config.py to set correct model paths
MODEL_PATH = 'path/to/your/model.pth'
FULL_MODEL_PATH = 'path/to/your/full_model.pth'
```

## 🚀 Usage

### API Server
```bash
# Start the FastAPI server
python main.py
# or
python api.py
```

### API Endpoints

#### Predict Emotion
```http
POST /action
Content-Type: application/json

{
    "text": "I'm so excited about this new opportunity!"
}
```

**Response:**
```json
{
    "sentiment": "happy"
}
```

### Direct Model Usage
```python
from model_loader import load_model, ModelPredictor
from preprocessing import basic_text_cleaning

# Load the model
model = load_model()
predictor = ModelPredictor(model)

# Predict emotion
text = "I'm feeling really anxious about tomorrow's presentation"
cleaned_text = basic_text_cleaning(text)
emotion = predictor.predict(cleaned_text)
print(f"Predicted emotion: {emotion}")
```

## 📁 Project Structure

```
Sentimental-Analysis/
├── 📄 main.py                 # Primary FastAPI application
├── 📄 api.py                  # Alternative API implementation
├── 📄 model_loader.py         # Model architecture and loading
├── 📄 predictor.py            # Prediction pipeline
├── 📄 preprocessing.py        # Text preprocessing utilities
├── 📄 config.py              # Configuration settings
├── 📄 schemas.py             # Pydantic data models
├── 📄 test_torch.py          # PyTorch environment testing
├── 📄 requirements.txt       # Python dependencies
├── 📓 MB.ipynb              # Main development notebook
├── 📓 ModelTesting.ipynb    # Model testing and evaluation
└── 📄 README.md             # Project documentation
```

## 🔧 Technical Implementation Details

### Key Technologies Used
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **ModernBERT**: Advanced transformer architecture
- **FastAPI**: Modern web framework for APIs
- **Pydantic**: Data validation and serialization
- **Dill**: Advanced pickle for model serialization
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing

### Advanced Features
- **Dynamic Text Processing**: Handles variable-length input sequences
- **Memory Optimization**: Efficient model loading and inference
- **Error Handling**: Comprehensive exception management
- **CORS Support**: Cross-origin resource sharing for web integration
- **Modular Design**: Clean separation of concerns

## 🧪 Testing & Validation

### Testing Methodology
- **Batch Testing**: Processed multiple text samples in batches
- **Cross-Validation**: Ensured model generalization
- **Performance Profiling**: Measured inference time and accuracy
- **Edge Case Testing**: Validated with extreme text lengths

### Test Results Documentation
- Comprehensive testing results saved in `outputs.csv`
- Performance metrics tracked across different text lengths
- Confusion matrix analysis for each emotion category

## 🚧 Future Enhancements

### Planned Improvements
1. **Dataset Quality Enhancement**
   - Collect higher-quality, more representative samples
   - Improve emotion labeling accuracy
   - Expand dataset size for better generalization

2. **Model Optimization**
   - Fine-tune ModernBERT for better accuracy
   - Implement ensemble methods
   - Add confidence scoring for predictions

3. **Production Deployment**
   - Docker containerization
   - Cloud deployment (AWS/GCP/Azure)
   - Load balancing and scaling

4. **Additional Features**
   - Real-time streaming analysis
   - Multi-language support
   - Emotion intensity scoring

## 🎓 Learning Outcomes

This project provided hands-on experience with:

- **Natural Language Processing (NLP)**
- **Deep Learning & Neural Networks**
- **Transformer Architecture**
- **Transfer Learning Techniques**
- **Generative AI & LLM Integration**
- **LangChain Framework**
- **FastAPI Development**
- **Model Deployment & Serving**
- **Data Preprocessing & Augmentation**
- **Performance Optimization**

## 📈 Business Impact

This sentiment analysis system can be applied to:
- **Social Media Monitoring**: Analyze user sentiment across platforms
- **Customer Service**: Automatically categorize support tickets
- **Content Moderation**: Detect harmful or inappropriate content
- **Market Research**: Understand customer emotions and preferences
- **Mental Health Applications**: Early detection of concerning emotional patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- New features
- Documentation enhancements

## 📄 License

This project is developed as part of an internship program. Please contact the author for licensing information.

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through the project repository.

---

**Built with ❤️ using ModernBERT and FastAPI**
