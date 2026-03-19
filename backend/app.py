from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def load_models():
    """Load BERT model and tokenizer"""
    global model, tokenizer
    
    print("=" * 80)
    print("LOADING BERT MODEL")
    print("=" * 80)
    
    try:
        model_path = 'backend/models/bert_sentiment_model'
        
        if os.path.exists(model_path):
            print(f"\n📂 Loading BERT from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()  # Set to evaluation mode
            print("✅ BERT model loaded successfully!")
        else:
            print(f"❌ Model not found at: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    print("=" * 80 + "\n")
    return model, tokenizer

# Load models on startup
model, tokenizer = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or tokenizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Tokenize
        inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=512)
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        pred = torch.argmax(logits, dim=1).item()
        prob = probs[0][pred].item()
        sentiment = 'Positive' if pred == 1 else 'Negative'
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(prob),
            'review': review
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
