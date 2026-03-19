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
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build paths relative to this file
    model_path = os.path.join(current_dir, 'model.pkl')
    vectorizer_path = os.path.join(current_dir, 'vectorizer.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from: {model_path}")
    except FileNotFoundError:
        print(f"❌ Model not found at: {model_path}")
        print(f"Available files in {current_dir}:")
        try:
            files = os.listdir(current_dir)
            for f in files:
                print(f"  - {f}")
        except:
            pass
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded from: {vectorizer_path}")
    except FileNotFoundError:
        print(f"❌ Vectorizer not found at: {vectorizer_path}")
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")

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
            return jsonify({'error': 'Models not loaded. Check server logs.'}), 500
        
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
