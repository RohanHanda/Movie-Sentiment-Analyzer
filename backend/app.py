from flask import Flask, request, jsonify
import joblib
import os
import sys

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
    print(f"Current directory: {os.getcwd()}")
    print("=" * 60)
    
    try:
        # Try multiple locations
        search_paths = [
            '/app',
            os.getcwd(),
            os.path.join(os.getcwd(), 'backend'),
            '/app/backend'
        ]
        
        model_file = None
        vectorizer_file = None
        
        for search_path in search_paths:
            if os.path.exists(search_path):
                print(f"\n📂 Checking: {search_path}")
                files = os.listdir(search_path)
                print(f"   Files: {files}")
                
                for file in files:
                    if 'sentiment_model' in file and file.endswith('.joblib'):
                        model_file = os.path.join(search_path, file)
                    if 'tfidf_vectorizer' in file and file.endswith('.joblib'):
                        vectorizer_file = os.path.join(search_path, file)
        
        if model_file and vectorizer_file:
            print(f"\n✅ Loading model from: {model_file}")
            print(f"✅ Loading vectorizer from: {vectorizer_file}")
            model = joblib.load(model_file)
            vectorizer = joblib.load(vectorizer_file)
            print("✅ Models loaded successfully!")
        else:
            print(f"\n❌ Model files not found!")
            print(f"   model_file: {model_file}")
            print(f"   vectorizer_file: {vectorizer_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

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
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
