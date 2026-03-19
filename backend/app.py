from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

app = Flask(__name__)
CORS(app)

model = None
vectorizer = None

def train_model_on_startup():
    """Train model on app startup using IMDB dataset"""
    global model, vectorizer
    
    print("=" * 80)
    print("TRAINING MODEL ON STARTUP")
    print("=" * 80)
    
    try:
        import kagglehub
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        
        print("\n📥 Downloading IMDB dataset from Kaggle...")
        path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
        csv_path = os.path.join(path, "IMDB Dataset.csv")
        
        print("📂 Loading CSV...")
        df = pd.read_csv(csv_path)
        
        # Prepare data
        df.columns = df.columns.str.lower()
        df['label'] = (df['sentiment'] == 'positive').astype(int)
        
        reviews = df['review'].values
        labels = df['label'].values
        
        print(f"\n📊 Dataset: {len(reviews)} reviews")
        print(f"  Positive: {sum(labels == 1)}")
        print(f"  Negative: {sum(labels == 0)}")
        
        # Split data
        print("\n🔀 Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            reviews, labels, test_size=0.2, random_state=42
        )
        
        # Train vectorizer
        print("📊 Training TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=5,
            max_df=0.7,
            stop_words='english',
            ngram_range=(1, 3),
            sublinear_tf=True,
            lowercase=True
        )
        
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        print(f"✅ Vectorizer ready! Vocab: {len(vectorizer.vocabulary_)}")
        
        # Train Random Forest
        print("🤖 Training Random Forest Model...")
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        from sklearn.metrics import accuracy_score
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model trained! Accuracy: {accuracy*100:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Using lightweight model instead...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        
        reviews = [
            "This movie is absolutely amazing! I loved every minute of it.",
            "Terrible waste of time. Horrible acting and plot.",
            "Fantastic cinematography and great acting.",
            "Awful movie. I fell asleep halfway through.",
            "Brilliant storytelling with excellent performances.",
            "Poorly directed and confusing plot.",
            "Outstanding! One of the best movies I've ever seen.",
            "Unwatchable garbage. Worst movie ever made.",
        ]
        labels = [1, 0, 1, 0, 1, 0, 1, 0]
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        X = vectorizer.fit_transform(reviews)
        
        model = MultinomialNB()
        model.fit(X, labels)
        
        print("✅ Fallback model ready")
    
    print("=" * 80 + "\n")
    return model, vectorizer

# Train on startup
model, vectorizer = train_model_on_startup()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        review = data.get('review', '').strip()
        
        if not review:
            return jsonify({'error': 'Review cannot be empty'}), 400
        
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
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
