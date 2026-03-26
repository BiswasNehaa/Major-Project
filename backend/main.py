from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import io

app = FastAPI()
analyzer = SentimentIntensityAnalyzer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/analyze")
async def analyze_sentiment(text: str):
    score = analyzer.polarity_scores(text)
    compound = score['compound']
    sentiment = "Positive" if compound >= 0.05 else "Negative" if compound <= -0.05 else "Neutral"
    return {"sentiment": sentiment, "score": abs(compound)}

# --- NEW BULK UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Read the CSV file
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Check if 'review_text' column exists
    if 'review_text' not in df.columns:
        return {"error": "CSV must have a 'review_text' column"}

    # Analyze every row
    results = []
    for text in df['review_text']:
        score = analyzer.polarity_scores(str(text))['compound']
        results.append(score)
    
    # Calculate Averages
    avg_score = sum(results) / len(results)
    sentiment = "Positive" if avg_score > 0 else "Negative"
    
    return {
        "total_reviews": len(results),
        "average_score": abs(round(avg_score, 2)),
        "sentiment": sentiment
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)