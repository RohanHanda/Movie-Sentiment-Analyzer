from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load models - fix the path
model = None
vectorizer = None

try:
    # Try multiple possible paths
    model_paths = [
        'backend/model.pkl',
        'model.pkl',
        '/app/backend/model.pkl',
        '/app/model.pkl'
    ]
    
    vectorizer_paths = [
        'backend/vectorizer.pkl',
        'vectorizer.pkl',
        '/app/backend/vectorizer.pkl',
        '/app/vectorizer.pkl'
    ]
    
    # Try to load model
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"✅ Model loaded from: {path}")
            break
    
    # Try to load vectorizer
    for path in vectorizer_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"✅ Vectorizer loaded from: {path}")
            break
    
    if model is None:
        print("❌ Model not found!")
    if vectorizer is None:
        print("❌ Vectorizer not found!")
        
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        review_vec = vectorizer.transform([review])
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
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
