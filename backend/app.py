from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import sys

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    
    print("=" * 50)
    print("DEBUG: Starting model loading...")
    print(f"Current working directory: {os.getcwd()}")
    print("=" * 50)
    
    # List all directories
    print("\nListing root directory:")
    try:
        for item in os.listdir('.'):
            full_path = os.path.join('.', item)
            if os.path.isdir(full_path):
                print(f"  📁 {item}/")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # List backend directory
    print("\nListing backend directory:")
    try:
        for item in os.listdir('./backend'):
            print(f"  - {item}")
    except Exception as e:
        print(f"Error listing backend: {e}")
    
    # Try loading model
    paths_to_try = [
        './backend/model.pkl',
        'backend/model.pkl',
        '/app/backend/model.pkl',
    ]
    
    print("\nTrying to load model from:")
    for path in paths_to_try:
        print(f"  Trying: {path}")
        if os.path.exists(path):
            print(f"    ✅ Found!")
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                print(f"    ✅ Successfully loaded model")
                break
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
        else:
            print(f"    ❌ Not found")
    
    # Try loading vectorizer
    print("\nTrying to load vectorizer from:")
    paths_to_try = [
        './backend/vectorizer.pkl',
        'backend/vectorizer.pkl',
        '/app/backend/vectorizer.pkl',
    ]
    
    for path in paths_to_try:
        print(f"  Trying: {path}")
        if os.path.exists(path):
            print(f"    ✅ Found!")
            try:
                with open(path, 'rb') as f:
                    vectorizer = pickle.load(f)
                print(f"    ✅ Successfully loaded vectorizer")
                break
            except Exception as e:
                print(f"    ❌ Error loading: {e}")
        else:
            print(f"    ❌ Not found")
    
    print("=" * 50)
    print(f"Status: model={model is not None}, vectorizer={vectorizer is not None}")
    print("=" * 50)

load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Models not loaded'}), 500
        
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
