from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def load_models():
    """Load pre-trained LightGBM models"""
    global model, vectorizer
    
    print("=" * 80)
    print("LOADING LIGHTGBM MODELS")
    print("=" * 80)
    
    try:
        print("\n📂 Loading model files...")
        print(f"   Current directory: {os.getcwd()}")
        
        # Try multiple paths
        model_paths = [
            'sentiment_model.joblib',
            '/app/sentiment_model.joblib',
            './sentiment_model.joblib'
        ]
        
        vectorizer_paths = [
            'tfidf_vectorizer.joblib',
            '/app/tfidf_vectorizer.joblib',
            './tfidf_vectorizer.joblib'
        ]
        
        # Load model
        for path in model_paths:
            if os.path.exists(path):
                print(f"   Found model at: {path}")
                model = joblib.load(path)
                print("   ✅ Model loaded!")
                break
        
        if model is None:
            print("   ❌ Model not found!")
            print(f"   Available files: {os.listdir('.')}")
        
        # Load vectorizer
        for path in vectorizer_paths:
            if os.path.exists(path):
                print(f"   Found vectorizer at: {path}")
                vectorizer = joblib.load(path)
                print("   ✅ Vectorizer loaded!")
                break
        
        if vectorizer is None:
            print("   ❌ Vectorizer not found!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
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
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Transform review
        review_vec = vectorizer.transform([review])
        
        # Make prediction
        pred = model.predict(review_vec)[0]
        prob = model.predict_proba(review_vec)[0].max()
        sentiment = 'Positive' if pred == 1 else 'Negative'
        
        return jsonify({
            'sentiment': sentiment,
            'confidence': float(prob),
            'review': review
        })
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })

@app.route('/info', methods=['GET'])
def info():
    """Get model information"""
    return jsonify({
        'model': 'LightGBM',
        'accuracy': '83.76%',
        'speed': 'Fast',
        'model_size': '50MB'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
