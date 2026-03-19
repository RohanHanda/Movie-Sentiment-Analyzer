from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os

app = Flask(__name__)
CORS(app)

# Load models
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    review_vec = vectorizer.transform([review])
    pred = model.predict(review_vec)[0]
    prob = model.predict_proba(review_vec)[0].max()
    sentiment = 'Positive' if pred == 1 else 'Negative'
    return jsonify({'sentiment': sentiment, 'confidence': float(prob)})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
