import React, { useState } from 'react';
import './App.css';

function App() {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);

  const analyze = async () => {
    const res = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review })
    });
    const data = await res.json();
    setResult(data);
  };

  return (
    <div className="app">
      <h1>Movie Sentiment Analyzer</h1>
      <textarea
        value={review}
        onChange={(e) => setReview(e.target.value)}
        placeholder="Enter movie review..."
        rows={5}
        cols={50}
      />
      <br />
      <button onClick={analyze}>Analyze</button>
      {result && (
        <div className={`result ${result.sentiment.toLowerCase()}`}>
          Sentiment: {result.sentiment} ({(result.confidence * 100).toFixed(1)}% confidence)
        </div>
      )}
    </div>
  );
}

export default App;
