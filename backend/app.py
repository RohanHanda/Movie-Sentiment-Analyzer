from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

model = None
vectorizer = None

# Manual CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.before_request
def handle_preflight():
    if request.method == 'OPTIONS':
        return '', 204

def load_models():
    global model, vectorizer
    print("Loading models...")
    
    try:
        model = joblib.load('backend/sentiment_model.joblib')
        vectorizer = joblib.load('backend/tfidf_vectorizer.joblib')
        print("✅ Models loaded!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

load_models()

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
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
