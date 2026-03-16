from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import uvicorn

# 1. Initialize the App
app = FastAPI()

# 2. Enable CORS (Crucial for React communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your React app to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Initialize AI Tools
analyzer = SentimentIntensityAnalyzer()

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Pulse AI Backend is running"}

# Objective 3: Sentiment & Severity Scoring
@app.get("/analyze")
def analyze_review(text: str):
    """
    Takes a string and returns sentiment scores.
    Example: localhost:8000/analyze?text=I love this app!
    """
    scores = analyzer.polarity_scores(text)
    
    # Determine sentiment label
    if scores['compound'] >= 0.05:
        sentiment = "Positive"
    elif scores['compound'] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {
        "text": text,
        "sentiment": sentiment,
        "scores": scores
    }

# Objective 1: Topic Modeling (LDA) Placeholder
@app.get("/topics")
def get_topics():
    """
    This will eventually use Gensim to cluster reviews.
    For now, it returns dummy data to test your frontend charts.
    """
    return {
        "topics": [
            {"id": 1, "name": "User Interface", "count": 45},
            {"id": 2, "name": "Performance/Lag", "count": 30},
            {"id": 3, "name": "Customer Support", "count": 12}
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)