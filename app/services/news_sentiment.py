import os
import logging
from typing import Dict, Literal
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client with Replit AI Integrations
client = OpenAI(
    api_key=os.environ.get("AI_INTEGRATIONS_OPENAI_API_KEY"),
    base_url=os.environ.get("AI_INTEGRATIONS_OPENAI_BASE_URL")
)

class NewsSentimentAnalyzer:
    def __init__(self):
        self.client = client
        
    async def analyze_news_impact(
        self, 
        news_title: str, 
        news_source: str,
        currencies: list[str],
        votes: Dict
    ) -> Dict:
        """
        Use AI to analyze news sentiment and trading impact
        
        Returns:
            {
                'sentiment': 'bullish' | 'bearish' | 'neutral',
                'impact_score': 0-10,
                'confidence': 0-100,
                'reasoning': str,
                'suggested_action': 'LONG' | 'SHORT' | 'WAIT',
                'affected_coins': List[str]
            }
        """
        try:
            # Calculate community sentiment from votes
            positive_votes = votes.get('positive', 0)
            negative_votes = votes.get('negative', 0)
            important_votes = votes.get('important', 0)
            
            vote_sentiment = "neutral"
            if positive_votes > negative_votes * 1.5:
                vote_sentiment = "bullish"
            elif negative_votes > positive_votes * 1.5:
                vote_sentiment = "bearish"
            
            # Build analysis prompt
            prompt = f"""You are a STRICT cryptocurrency trading analyst. Only flag EXTREMELY HIGH IMPACT news that will definitely move markets.

News Title: {news_title}
Source: {news_source}
Mentioned Coins: {', '.join(currencies) if currencies else 'General crypto market'}
Community Votes: {positive_votes} positive, {negative_votes} negative, {important_votes} important

STRICT CRITERIA - Only give high scores (9-10) and high confidence (80%+) for:
- Major institutional moves (ETF approvals, large corporate purchases)
- Critical regulatory news (SEC decisions, government bans/approvals)
- Major protocol/network events (hard forks, critical bugs, major upgrades)
- Massive market events (exchange hacks, major liquidations)

DO NOT score high for:
- General price predictions or analysis
- Small partnership announcements
- Minor technical updates
- Routine market commentary
- Speculation or rumors

Analyze and provide:
1. Overall sentiment (bullish/bearish/neutral)
2. Impact score (0-10, be VERY conservative - most news should be 5 or below)
3. Your confidence level (0-100%, require strong evidence for high confidence)
4. Brief reasoning (1-2 sentences)
5. Suggested trading action (LONG/SHORT/WAIT)
6. Which specific coins are most affected

Respond in this exact JSON format:
{{
    "sentiment": "bullish|bearish|neutral",
    "impact_score": 0-10,
    "confidence": 0-100,
    "reasoning": "your analysis here",
    "suggested_action": "LONG|SHORT|WAIT",
    "affected_coins": ["BTC", "ETH"]
}}"""

            # Call OpenAI API via Replit AI Integrations
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional crypto trading analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            # Parse AI response
            import json
            analysis = json.loads(response.choices[0].message.content)
            
            # Validate and normalize response
            result = {
                'sentiment': analysis.get('sentiment', 'neutral').lower(),
                'impact_score': min(10, max(0, int(analysis.get('impact_score', 0)))),
                'confidence': min(100, max(0, int(analysis.get('confidence', 0)))),
                'reasoning': analysis.get('reasoning', 'No reasoning provided'),
                'suggested_action': analysis.get('suggested_action', 'WAIT').upper(),
                'affected_coins': analysis.get('affected_coins', currencies)
            }
            
            # Apply vote-based adjustment
            if vote_sentiment == 'bullish' and result['sentiment'] == 'neutral':
                result['sentiment'] = 'bullish'
            elif vote_sentiment == 'bearish' and result['sentiment'] == 'neutral':
                result['sentiment'] = 'bearish'
            
            logger.info(f"News analysis: {result['sentiment']} ({result['impact_score']}/10) - {news_title[:50]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            # Fallback to vote-based analysis
            return self._fallback_analysis(news_title, votes, currencies)
    
    def _fallback_analysis(self, title: str, votes: Dict, currencies: list[str]) -> Dict:
        """Fallback analysis based on votes if AI fails"""
        positive = votes.get('positive', 0)
        negative = votes.get('negative', 0)
        important = votes.get('important', 0)
        
        # Determine sentiment from votes
        if positive > negative * 1.5:
            sentiment = 'bullish'
            action = 'LONG'
        elif negative > positive * 1.5:
            sentiment = 'bearish'
            action = 'SHORT'
        else:
            sentiment = 'neutral'
            action = 'WAIT'
        
        # Impact based on importance votes
        impact_score = min(10, important // 5)
        
        # Confidence based on total votes
        total_votes = positive + negative + important
        confidence = min(100, total_votes * 5)
        
        return {
            'sentiment': sentiment,
            'impact_score': impact_score,
            'confidence': confidence,
            'reasoning': f'Community-driven analysis: {positive} positive vs {negative} negative votes',
            'suggested_action': action,
            'affected_coins': currencies
        }
    
    def should_generate_signal(self, analysis: Dict) -> bool:
        """
        Determine if news warrants a trading signal
        
        Criteria (STRICT - only most important news):
        - Very high impact (score >= 9/10)
        - Very high confidence (>= 80%)
        - Clear directional bias (not neutral)
        """
        return (
            analysis['impact_score'] >= 9 and
            analysis['confidence'] >= 80 and
            analysis['sentiment'] != 'neutral' and
            analysis['suggested_action'] in ['LONG', 'SHORT']
        )
