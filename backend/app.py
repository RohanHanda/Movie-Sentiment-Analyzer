from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import subprocess

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def check_system_libs():
    """Check if required system libraries are available"""
    print("=" * 80)
    print("CHECKING SYSTEM LIBRARIES")
    print("=" * 80)
    
    libs = ['libgomp.so.1', 'libgomp.so']
    for lib in libs:
        try:
            result = subprocess.run(['ldconfig', '-p'], capture_output=True, text=True)
            if lib in result.stdout:
                print(f"✅ Found: {lib}")
            else:
                print(f"❌ Missing: {lib}")
        except Exception as e:
            print(f"❌ Error checking {lib}: {e}")
    print("=" * 80 + "\n")

def load_models():
    """Load pre-trained models using joblib"""
    global model, vectorizer
    
    check_system_libs()
    
    print("=" * 80)
    print("LOADING MODELS WITH JOBLIB")
    print("=" * 80)
    
    model_paths = [
        'backend/sentiment_model.joblib',
        './backend/sentiment_model.joblib',
        '/app/backend/sentiment_model.joblib',
    ]
    
    vectorizer_paths = [
        'backend/tfidf_vectorizer.joblib',
        './backend/tfidf_vectorizer.joblib',
        '/app/backend/tfidf_vectorizer.joblib',
    ]
    
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
        'vectorizer_loaded': vectorizer is not None,
        'cwd': os.getcwd()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
