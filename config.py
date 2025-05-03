import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = r'C:\Users\Dell\Downloads\Data Science Stuff\Machine Learning\Internship Tasks\Sentimental Analysis Model\model.pth'
MODEL_NAME = "answerdotai/ModernBERT-base"
EMOTION_LABELS = ['Aggressive','Happy','Sad','Neutral','Fear']
FULL_MODEL_PATH = r'C:\Users\Dell\Downloads\Data Science Stuff\Machine Learning\Internship Tasks\Sentimental Analysis Model\full_model.pth'
