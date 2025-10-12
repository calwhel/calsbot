import logging
import ccxt.async_support as ccxt
from typing import Dict, List, Optional
from datetime import datetime
from app.services.news_monitor import NewsMonitor
from app.services.news_sentiment import NewsSentimentAnalyzer
from app.config import settings

logger = logging.getLogger(__name__)

class NewsSignalGenerator:
    def __init__(self):
        self.news_monitor = NewsMonitor()
        self.sentiment_analyzer = NewsSentimentAnalyzer()
        self.exchange = None
        
    async def initialize_exchange(self):
        """Initialize exchange for price data"""
        if not self.exchange:
            exchange_id = settings.EXCHANGE.lower()
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class({
                'enableRateLimit': True,
            })
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol"""
        try:
            await self.initialize_exchange()
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last']
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def scan_news_for_signals(self, trading_symbols: List[str]) -> List[Dict]:
        """
        Scan news and generate trading signals
        
        Returns list of news-based signals with format:
        {
            'type': 'news',
            'symbol': str,
            'direction': 'LONG' | 'SHORT',
            'entry_price': float,
            'news_title': str,
            'news_url': str,
            'sentiment': str,
            'impact_score': int,
            'confidence': int,
            'reasoning': str,
            'timestamp': datetime
        }
        """
        signals = []
        
        try:
            # Get important news for our trading symbols
            news_articles = await self.news_monitor.get_important_news(trading_symbols)
            
            if not news_articles:
                logger.info("No new important news found")
                return signals
            
            logger.info(f"Analyzing {len(news_articles)} news articles...")
            
            for article in news_articles:
                metadata = self.news_monitor.get_news_metadata(article)
                
                # Analyze sentiment with AI
                analysis = await self.sentiment_analyzer.analyze_news_impact(
                    news_title=metadata['title'],
                    news_source=metadata['source'],
                    currencies=metadata['currencies'],
                    votes=metadata['votes']
                )
                
                # Check if this news warrants a signal
                if not self.sentiment_analyzer.should_generate_signal(analysis):
                    logger.info(f"News doesn't meet signal criteria: {metadata['title'][:50]}...")
                    continue
                
                # Generate signals for affected coins
                affected_coins = analysis['affected_coins']
                
                for coin in affected_coins:
                    # Find matching trading symbol
                    matching_symbols = [s for s in trading_symbols if s.startswith(coin + '/')]
                    
                    for symbol in matching_symbols:
                        # Get current price
                        current_price = await self.get_current_price(symbol)
                        
                        if not current_price:
                            continue
                        
                        # Determine direction based on sentiment
                        direction = analysis['suggested_action']
                        
                        # Calculate stop loss and take profit based on impact
                        # Higher impact = wider stops
                        impact_multiplier = analysis['impact_score'] / 10
                        stop_distance = current_price * 0.02 * impact_multiplier  # 2% base * impact
                        tp_distance = current_price * 0.05 * impact_multiplier    # 5% base * impact
                        
                        if direction == 'LONG':
                            stop_loss = current_price - stop_distance
                            take_profit = current_price + tp_distance
                        else:  # SHORT
                            stop_loss = current_price + stop_distance
                            take_profit = current_price - tp_distance
                        
                        signal = {
                            'type': 'news',
                            'symbol': symbol,
                            'direction': direction,
                            'entry_price': current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'news_title': metadata['title'],
                            'news_url': metadata['url'],
                            'news_source': metadata['source'],
                            'sentiment': analysis['sentiment'],
                            'impact_score': analysis['impact_score'],
                            'confidence': analysis['confidence'],
                            'reasoning': analysis['reasoning'],
                            'timestamp': datetime.utcnow(),
                            'timeframe': 'NEWS'
                        }
                        
                        signals.append(signal)
                        logger.info(f"📰 News signal generated: {direction} {symbol} (Impact: {analysis['impact_score']}/10)")
            
            # Clean up old seen IDs periodically
            self.news_monitor.clean_old_seen_ids()
            
        except Exception as e:
            logger.error(f"Error scanning news for signals: {e}")
        finally:
            if self.exchange:
                await self.exchange.close()
                self.exchange = None
        
        return signals
    
    def format_news_signal_message(self, signal: Dict) -> str:
        """Format news signal for broadcast"""
        direction_emoji = "🚀" if signal['direction'] == 'LONG' else "📉"
        
        # Calculate potential PnL with 10x leverage
        entry = signal['entry_price']
        tp = signal['take_profit']
        sl = signal['stop_loss']
        
        if signal['direction'] == 'LONG':
            tp_pnl = ((tp - entry) / entry) * 100 * 10
            sl_pnl = ((sl - entry) / entry) * 100 * 10
        else:
            tp_pnl = ((entry - tp) / entry) * 100 * 10
            sl_pnl = ((entry - sl) / entry) * 100 * 10
        
        message = f"""
{direction_emoji} **NEWS-BASED SIGNAL**

📰 **Breaking News Impact**
{signal['news_title']}

📊 **Trade Setup**
Symbol: {signal['symbol']}
Direction: {signal['direction']}
Entry: ${signal['entry_price']:.4f}

🎯 Targets:
• TP: ${signal['take_profit']:.4f} ({'+' if tp_pnl > 0 else ''}{tp_pnl:.1f}% @ 10x)
• SL: ${signal['stop_loss']:.4f} ({'+' if sl_pnl > 0 else ''}{sl_pnl:.1f}% @ 10x)

📈 **News Analysis**
Sentiment: {signal['sentiment'].upper()}
Impact Score: {signal['impact_score']}/10
Confidence: {signal['confidence']}%

💡 {signal['reasoning']}

🔗 Source: {signal['news_source']}
📰 Read: {signal['news_url']}

⚠️ News-based signals can be volatile. Use proper risk management!
"""
        return message
