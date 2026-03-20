from flask import Flask, request, jsonify, make_response
import joblib
import os

app = Flask(__name__)

model = None
vectorizer = None

# Manually add CORS headers to every response
@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '3600'
    return response

def load_models():
    """Load pre-trained models using joblib"""
    global model, vectorizer
    
    print("=" * 80)
    print("LOADING MODELS WITH JOBLIB")
    print("=" * 80)
    
    model_paths = [
        'sentiment_model.joblib',
        '/app/sentiment_model.joblib',
        'backend/sentiment_model.joblib',
    ]
    
    vectorizer_paths = [
        'tfidf_vectorizer.joblib',
        '/app/tfidf_vectorizer.joblib',
        'backend/tfidf_vectorizer.joblib',
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
    
    if vectorizer is None:
        print("  ⚠️  Vectorizer not found at any path!")
    
    print("\n" + "=" * 80)
    print(f"Status: model_loaded={model is not None}, vectorizer_loaded={vectorizer is not None}")
    print("=" * 80 + "\n")

# Load models on startup
load_models()

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """Predict sentiment of a movie review"""
    # Handle preflight request
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        return response
    
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
        print(f"❌ Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET', 'OPTIONS'])
def health():
    """Health check endpoint"""
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        return response
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'cwd': os.getcwd()
    })

@app.route('/', methods=['GET', 'OPTIONS'])
def index():
    """Root endpoint"""
    if request.method == 'OPTIONS':
        response = make_response('', 204)
        return response
    
    return jsonify({
        'message': 'Movie Sentiment Analyzer API',
        'endpoints': {
            'POST /predict': 'Predict sentiment',
            'GET /health': 'Health check'
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
