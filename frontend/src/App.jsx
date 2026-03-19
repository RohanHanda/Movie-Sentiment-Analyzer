import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import API_BASE_URL from './config';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [sentiment, setSentiment] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSentiment(null);
    setConfidence(null);

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ review }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setSentiment(data.sentiment);
      setConfidence((data.confidence * 100).toFixed(2));
      setReview('');
    } catch (err) {
      setError(err.message || 'Failed to analyze sentiment. Make sure backend is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Router basename="/Movie-Sentiment-Analyzer">
      <Routes>
        <Route
          path="/"
          element={
            <div className="container">
              <h1>🎬 Movie Sentiment Analyzer</h1>
              <p>Enter a movie review to analyze its sentiment</p>

              <form onSubmit={handleSubmit}>
                <textarea
                  value={review}
                  onChange={(e) => setReview(e.target.value)}
                  placeholder="Enter your movie review here..."
                  required
                  disabled={loading}
                />
                <button type="submit" disabled={loading || !review.trim()}>
                  {loading ? 'Analyzing...' : 'Analyze Sentiment'}
                </button>
              </form>

              {error && <div className="error">{error}</div>}

              {sentiment && (
                <div className="result">
                  <h2>Result:</h2>
                  <p className={`sentiment ${sentiment.toLowerCase()}`}>
                    Sentiment: <strong>{sentiment}</strong>
                  </p>
                  <p>Confidence: <strong>{confidence}%</strong></p>
                </div>
              )}
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

export default App;
