from typing import Dict
import nltk

def _ensure_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # noqa
        nltk.data.find('sentiment/vader_lexicon.zip')
    except Exception:
        nltk.download('vader_lexicon', quiet=True)

_ensure_vader()
from nltk.sentiment import SentimentIntensityAnalyzer

class EmotionAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, float]:
        scores = self.analyzer.polarity_scores(text or "")
        mood = "neutral"
        if scores["compound"] >= 0.35:
            mood = "positive"
        elif scores["compound"] <= -0.35:
            mood = "negative"
        scores["mood"] = mood
        return scores
