"""
MarketAux API Client - Fetches gold/silver news with sentiment analysis.

Used for metals trading signals based on news-driven market moves.
"""

import os
import logging
import httpx
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

MARKETAUX_BASE_URL = "https://api.marketaux.com/v1"


class MarketAuxClient:
    """Client for MarketAux financial news API."""
    
    def __init__(self):
        self.api_key = os.environ.get("MARKETAUX_API_KEY")
        if not self.api_key:
            logger.warning("MARKETAUX_API_KEY not set")
    
    async def get_metals_news(self, symbols: List[str] = None, limit: int = 10) -> Optional[List[Dict]]:
        """Fetch latest news for gold/silver with sentiment.
        
        Args:
            symbols: List of symbols like ["XAU", "XAG", "GOLD", "SILVER"]
            limit: Max articles to return
            
        Returns:
            List of articles with sentiment data
        """
        if not self.api_key:
            logger.error("MarketAux API key not configured")
            return None
        
        if symbols is None:
            symbols = ["XAU", "XAG", "GOLD", "SILVER"]
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                params = {
                    "api_token": self.api_key,
                    "symbols": ",".join(symbols),
                    "filter_entities": "true",
                    "language": "en",
                    "limit": limit,
                    "published_after": (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M")
                }
                
                response = await client.get(f"{MARKETAUX_BASE_URL}/news/all", params=params)
                
                if response.status_code != 200:
                    logger.error(f"MarketAux API error: {response.status_code} - {response.text}")
                    return None
                
                data = response.json()
                
                if not data.get("data"):
                    logger.info("No news articles found for metals")
                    return []
                
                return data["data"]
                
        except Exception as e:
            logger.error(f"MarketAux API error: {e}")
            return None
    
    async def analyze_metals_sentiment(self) -> Dict:
        """Analyze overall sentiment for gold and silver.
        
        Returns:
            Dict with sentiment analysis:
            {
                "gold": {"sentiment": "bullish/bearish/neutral", "score": 0.5, "articles": 5},
                "silver": {"sentiment": "bullish/bearish/neutral", "score": 0.3, "articles": 3},
                "headlines": ["Top headline 1", "Top headline 2"],
                "recommendation": "LONG/SHORT/SKIP"
            }
        """
        articles = await self.get_metals_news(limit=20)
        
        if not articles:
            return {
                "gold": {"sentiment": "neutral", "score": 0, "articles": 0},
                "silver": {"sentiment": "neutral", "score": 0, "articles": 0},
                "headlines": [],
                "recommendation": "SKIP",
                "reason": "No recent news available"
            }
        
        gold_scores = []
        silver_scores = []
        headlines = []
        
        for article in articles:
            title = article.get("title", "")
            headlines.append(title)
            
            entities = article.get("entities", [])
            for entity in entities:
                symbol = entity.get("symbol", "").upper()
                sentiment_score = entity.get("sentiment_score", 0)
                
                if symbol in ["XAU", "GOLD"]:
                    gold_scores.append(sentiment_score)
                elif symbol in ["XAG", "SILVER"]:
                    silver_scores.append(sentiment_score)
        
        def calc_sentiment(scores):
            if not scores:
                return {"sentiment": "neutral", "score": 0, "articles": 0}
            avg = sum(scores) / len(scores)
            if avg > 0.2:
                sentiment = "bullish"
            elif avg < -0.2:
                sentiment = "bearish"
            else:
                sentiment = "neutral"
            return {"sentiment": sentiment, "score": round(avg, 3), "articles": len(scores)}
        
        gold_analysis = calc_sentiment(gold_scores)
        silver_analysis = calc_sentiment(silver_scores)
        
        combined_score = 0
        total = 0
        if gold_scores:
            combined_score += sum(gold_scores)
            total += len(gold_scores)
        if silver_scores:
            combined_score += sum(silver_scores)
            total += len(silver_scores)
        
        avg_combined = combined_score / total if total > 0 else 0
        
        if avg_combined > 0.3:
            recommendation = "LONG"
            reason = "Strong bullish sentiment in metals news"
        elif avg_combined < -0.3:
            recommendation = "SHORT"
            reason = "Strong bearish sentiment in metals news"
        else:
            recommendation = "SKIP"
            reason = "Mixed or neutral sentiment - no clear direction"
        
        return {
            "gold": gold_analysis,
            "silver": silver_analysis,
            "headlines": headlines[:5],
            "recommendation": recommendation,
            "reason": reason,
            "combined_score": round(avg_combined, 3),
            "total_articles": len(articles)
        }


async def get_metals_sentiment() -> Dict:
    """Convenience function to get metals sentiment analysis."""
    client = MarketAuxClient()
    return await client.analyze_metals_sentiment()
