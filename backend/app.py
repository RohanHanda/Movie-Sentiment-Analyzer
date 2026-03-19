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
    
    print("=" * 80)
    print("LOADING MODELS - DEBUG INFO")
    print("=" * 80)
    
    try:
        with open('backend/model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        with open('backend/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        print("✅ Vectorizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)
    print(f"FINAL STATUS: model={model is not None}, vectorizer={vectorizer is not None}")
    print("=" * 80 + "\n")

load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "=" * 80)
        print("PREDICT REQUEST RECEIVED")
        print("=" * 80)
        
        data = request.json
        print(f"Data received: {data}")
        
        review = data.get('review', '').strip()
        print(f"Review: {review}")
        
        if not review:
            print("❌ Review is empty")
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None:
            print("❌ Model is None")
            return jsonify({'error': 'Model not loaded'}), 500
        
        if vectorizer is None:
            print("❌ Vectorizer is None")
            return jsonify({'error': 'Vectorizer not loaded'}), 500
        
        print(f"🔄 Transforming review...")
        review_vec = vectorizer.transform([review])
        print(f"✅ Review transformed")
        
        print(f"🔄 Making prediction...")
        pred = model.predict(review_vec)[0]
        print(f"✅ Prediction: {pred}")
        
        prob = model.predict_proba(review_vec)[0].max()
        print(f"✅ Probability: {prob}")
        
        sentiment = 'Positive' if pred == 1 else 'Negative'
        print(f"✅ Sentiment: {sentiment}")
        
        response = {
            'sentiment': sentiment,
            'confidence': float(prob),
            'review': review
        }
        print(f"Response: {response}")
        print("=" * 80 + "\n")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 80 + "\n")
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
