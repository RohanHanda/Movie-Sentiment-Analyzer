from flask import Flask, request, jsonify
import pickle
import os
app = Flask(__name__)
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
if __name__ == '__main__':
    app.run(debug=True, port=5000)
