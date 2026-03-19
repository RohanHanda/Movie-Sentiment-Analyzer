from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import sys
import base64

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    
    print("=" * 80)
    print("LOADING MODELS - DEBUG INFO")
    print("=" * 80)
    print(f"CWD: {os.getcwd()}")
    
    # Check if we're in /app directory
    if os.path.exists('/app/backend'):
        print("✅ Found /app/backend")
        os.chdir('/app')
    
    print(f"Current CWD after check: {os.getcwd()}")
    
    # List what we have
    print("\n📂 Root directory contents:")
    for item in os.listdir('.'):
        size = ""
        path = os.path.join('.', item)
        if os.path.isfile(path):
            size = f" ({os.path.getsize(path)} bytes)"
        print(f"  {'📁' if os.path.isdir(path) else '📄'} {item}{size}")
    
    print("\n📂 Backend directory contents:")
    if os.path.exists('backend'):
        for item in os.listdir('backend'):
            path = os.path.join('backend', item)
            size = ""
            if os.path.isfile(path):
                size = f" ({os.path.getsize(path)} bytes)"
            print(f"  {'📁' if os.path.isdir(path) else '📄'} {item}{size}")
    else:
        print("  ❌ backend directory not found!")
    
    # Try to load model
    model_path = 'backend/model.pkl'
    vectorizer_path = 'backend/vectorizer.pkl'
    
    print(f"\n🔍 Looking for model at: {model_path}")
    print(f"   Exists: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        try:
            print(f"   Loading...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"   ✅ Model loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🔍 Looking for vectorizer at: {vectorizer_path}")
    print(f"   Exists: {os.path.exists(vectorizer_path)}")
    
    if os.path.exists(vectorizer_path):
        try:
            print(f"   Loading...")
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"   ✅ Vectorizer loaded successfully!")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print(f"FINAL STATUS: model_loaded={model is not None}, vectorizer_loaded={vectorizer is not None}")
    print("=" * 80 + "\n")

# Load models on startup
load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Models not loaded',
                'model_loaded': model is not None,
                'vectorizer_loaded': vectorizer is not None
            }), 500
        
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
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

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
