from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)
CORS(app)

model = None
tokenizer = None

def load_models():
    """Load pre-trained DistilBERT model for sentiment analysis"""
    global model, tokenizer
    
    print("=" * 80)
    print("LOADING DISTILBERT MODEL")
    print("=" * 80)
    
    try:
        # Pre-trained DistilBERT model fine-tuned on SST-2 dataset
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        
        print(f"\n📥 Loading model: {model_name}")
        print("   This model is already trained for sentiment classification!")
        
        print("📖 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded!")
        
        print("🤖 Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()  # Set to evaluation mode
        print("✅ Model loaded!")
        
        print("\n📊 Model Info:")
        print("   - Base: DistilBERT (40% smaller than BERT)")
        print("   - Training Data: SST-2 (Stanford Sentiment Treebank)")
        print("   - Expected Accuracy: ~90%+")
        print("   - Inference Speed: ~100ms per review")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80 + "\n")

# Load models on startup
load_models()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment of a movie review"""
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or tokenizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Tokenize the review
        inputs = tokenizer(
            review, 
            return_tensors='pt', 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction
        pred = torch.argmax(logits, dim=1).item()
        prob = probs[0][pred].item()
        
        # Map to sentiment (0=negative, 1=positive)
        sentiment = 'Positive' if pred == 1 else 'Negative'
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(prob),
            'review': review
        })
        
    except Exception as e:
        print(f"❌ Error in predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'DistilBERT',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
        'model_type': 'DistilBERT',
        'task': 'Sentiment Analysis',
        'accuracy': '90%+',
        'base_model': 'DistilBERT',
        'training_dataset': 'SST-2 (Stanford Sentiment Treebank)',
        'max_sequence_length': 512,
        'inference_time': '~100ms'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
