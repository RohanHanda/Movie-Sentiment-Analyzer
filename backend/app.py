from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = None
vectorizer = None

# CORS Headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        return '', 204

def load_models():
    global model, vectorizer
    print("=" * 60)
    print("Loading LightGBM models...")
    print("=" * 60)
    
    try:
        # Try different paths
        paths = [
            'sentiment_model.joblib',
            '../sentiment_model.joblib',
            './sentiment_model.joblib'
        ]
        
        for path in paths:
            if os.path.exists(path):
                print(f"✅ Found model at: {path}")
                model = joblib.load(path)
                vectorizer = joblib.load(path.replace('sentiment_model', 'tfidf_vectorizer'))
                print("✅ Models loaded successfully!")
                return
        
        print("❌ Models not found!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

load_models()

@app.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Movie Sentiment Analyzer API', 'status': 'running'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
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
        print(f"Error: {e}")
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
