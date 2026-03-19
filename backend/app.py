from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def load_models():
    """Load pre-trained models using joblib"""
    global model, vectorizer
    
    print("=" * 80)
    print("LOADING MODELS WITH JOBLIB")
    print("=" * 80)
    
    # Try different possible paths
    model_paths = [
        'backend/model.joblib',
    ]
    
    vectorizer_paths = [
        'backend/vectorizer.joblib',
    ]
    
    # Load model
    print("\n🔍 Loading sentiment model...")
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"  Found at: {path}")
                model = joblib.load(path)
                print(f"  ✅ Model loaded successfully!")
                break
            except Exception as e:
                print(f"  ❌ Error loading from {path}: {e}")
        else:
            print(f"  ❌ Not found: {path}")
    
    if model is None:
        print("  ⚠️  Model not found at any path!")
    
    # Load vectorizer
    print("\n🔍 Loading TF-IDF vectorizer...")
    for path in vectorizer_paths:
        if os.path.exists(path):
            try:
                print(f"  Found at: {path}")
                vectorizer = joblib.load(path)
                print(f"  ✅ Vectorizer loaded successfully!")
                break
            except Exception as e:
                print(f"  ❌ Error loading from {path}: {e}")
        else:
            print(f"  ❌ Not found: {path}")
    
    if vectorizer is None:
        print("  ⚠️  Vectorizer not found at any path!")
    
    print("\n" + "=" * 80)
    print(f"Status: model_loaded={model is not None}, vectorizer_loaded={vectorizer is not None}")
    print("=" * 80 + "\n")
    
    return model, vectorizer

# Load models on startup
model, vectorizer = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if vectorizer is None:
            return jsonify({'error': 'Vectorizer not loaded'}), 500
        
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
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'cwd': os.getcwd()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
