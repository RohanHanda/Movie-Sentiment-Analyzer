from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def download_from_gdrive():
    """Download model from Google Drive"""
    print("📥 Downloading model from Google Drive...")
    
    # Share your model folder on Google Drive and get the ID
    folder_id = "https://drive.google.com/drive/folders/1ScAMqRRYTezqOH1Y04QJZ6w9AXW8ol7l?usp=sharing"  # Replace with your folder ID
    
    os.makedirs('backend/models/bert_sentiment_model', exist_ok=True)
    
    # Download files
    gdown.download_folder(
        f"https://drive.google.com/drive/folders/{folder_id}",
        output='backend/models/bert_sentiment_model',
        quiet=False
    )

def load_models():
    global model, tokenizer
    
    print("=" * 80)
    print("LOADING BERT MODEL")
    print("=" * 80)
    
    try:
        model_path = 'backend/models/bert_sentiment_model'
        
        # Download if not exists
        if not os.path.exists(model_path):
            download_from_gdrive()
        
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("🤖 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("✅ Model loaded!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 80 + "\n")
    return model, tokenizer

model, tokenizer = load_models()

# ... rest of the code same as above
