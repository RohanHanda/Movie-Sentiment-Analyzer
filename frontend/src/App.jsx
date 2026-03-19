import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

function App() {
  return (
    <Router basename="/Movie-Sentiment-Analyzer">
      <Routes>
        <Route path="/" element={<div>Welcome to Movie Sentiment Analyzer</div>} />
      </Routes>
    </Router>
  );
}

export default App;
