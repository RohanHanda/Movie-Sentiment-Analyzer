from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def create_and_train_model():
    """Create and train model inline instead of loading from pickle"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    
    print("=" * 80)
    print("TRAINING MODEL INLINE")
    print("=" * 80)
    
    # Training data
    reviews = [
        "This movie is amazing! I loved it!",
        "Terrible film, waste of time",
        "Absolutely fantastic, best movie ever",
        "Horrible, couldn't finish watching",
        "Great acting and amazing plot",
        "Bad story, poor production",
        "Brilliant! Highly recommend",
        "Awful, one of the worst movies I've seen",
        "Outstanding! One of the best movies ever",
        "Waste of time, very disappointing",
        "Excellent film, highly recommended",
        "Poor quality, don't bother watching",
        "Amazing storyline and great actors",
        "Boring and predictable plot",
        "Best movie I've ever seen",
        "Worst movie ever made",
        "Beautiful cinematography",
        "Terrible acting and direction",
        "Loved every minute of it",
        "Absolutely dreadful"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Create and train vectorizer
    print("📊 Training vectorizer...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X = vectorizer.fit_transform(reviews)
    print(f"✅ Vectorizer trained! Vocab size: {len(vectorizer.vocabulary_)}")
    
    # Create and train model
    print("🤖 Training model...")
    model = MultinomialNB()
    model.fit(X, labels)
    print(f"✅ Model trained!")
    
    # Test
    print("\n🧪 Testing model:")
    for review in ["This is great!", "This is terrible!"]:
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()
        sentiment = 'Positive' if pred == 1 else 'Negative'
        print(f"  '{review}' → {sentiment} ({prob*100:.1f}%)")
    
    print("=" * 80 + "\n")
    
    return model, vectorizer

# Initialize model on startup
print("Initializing sentiment analyzer...")
model, vectorizer = create_and_train_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "=" * 80)
        print("PREDICT REQUEST RECEIVED")
        print("=" * 80)
        
        data = request.json
        review = data.get('review', '').strip()
        
        print(f"Review: {review}")
        
        if not review:
            print("❌ Review is empty")
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or vectorizer is None:
            print("❌ Model or vectorizer not initialized")
            return jsonify({'error': 'Model not initialized'}), 500
        
        print("🔄 Transforming review...")
        review_vec = vectorizer.transform([review])
        print("✅ Review transformed")
        
        print("🔄 Making prediction...")
        pred = model.predict(review_vec)[0]
        prob = model.predict_proba(review_vec)[0].max()
        sentiment = 'Positive' if pred == 1 else 'Negative'
        
        print(f"✅ Sentiment: {sentiment} ({prob*100:.2f}%)")
        
        response = {
            'sentiment': sentiment,
            'confidence': float(prob),
            'review': review
        }
        
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
        'vectorizer_loaded': vectorizer is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
