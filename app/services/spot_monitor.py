import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt.async_support as ccxt
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import SpotActivity
import logging

logger = logging.getLogger(__name__)

class SpotMarketMonitor:
    def __init__(self):
        self.exchanges = {
            'coinbase': ccxt.coinbase(),
            'kraken': ccxt.kraken(),
            'okx': ccxt.okx()
        }
        self.failed_exchanges = set()
        self.exchange_symbols = {}
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 
            'ADA/USDT', 'DOGE/USDT', 'TRX/USDT', 'AVAX/USDT',
            'DOT/USDT', 'LINK/USDT', 'LTC/USDT', 'UNI/USDT',
            'ATOM/USDT', 'BCH/USDT', 'APT/USDT'
        ]
        
    async def load_exchange_symbols(self, exchange_id: str):
        """Load and cache available symbols for an exchange"""
        if exchange_id in self.failed_exchanges:
            return []
            
        if exchange_id in self.exchange_symbols:
            return self.exchange_symbols[exchange_id]
        
        try:
            exchange = self.exchanges[exchange_id]
            markets = await exchange.load_markets()
            available_symbols = [s for s in self.symbols if s in markets]
            self.exchange_symbols[exchange_id] = available_symbols
            logger.info(f"{exchange_id}: {len(available_symbols)} symbols available")
            return available_symbols
        except Exception as e:
            error_msg = str(e).lower()
            if '451' in str(e) or '403' in str(e) or 'restricted' in error_msg or 'forbidden' in error_msg:
                logger.warning(f"{exchange_id} is geo-restricted, disabling permanently")
                self.failed_exchanges.add(exchange_id)
            else:
                logger.error(f"Error loading markets for {exchange_id}: {e}")
            return []
        
    async def calculate_order_book_imbalance(self, exchange_id: str, symbol: str) -> Optional[float]:
        """Calculate bid/ask imbalance from order book (-1 to 1, positive = buying pressure)"""
        try:
            exchange = self.exchanges[exchange_id]
            orderbook = await exchange.fetch_order_book(symbol, limit=20)
            
            bid_volume = sum([bid[1] for bid in orderbook['bids'][:10]])
            ask_volume = sum([ask[1] for ask in orderbook['asks'][:10]])
            
            total_volume = bid_volume + ask_volume
            if total_volume == 0:
                return None
                
            imbalance = (bid_volume - ask_volume) / total_volume
            return float(imbalance)
            
        except Exception as e:
            logger.error(f"Error calculating imbalance for {exchange_id} {symbol}: {e}")
            return None
    
    async def get_recent_trades_pressure(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """Analyze recent trades to detect buying/selling pressure"""
        try:
            exchange = self.exchanges[exchange_id]
            trades = await exchange.fetch_trades(symbol, limit=100)
            
            if not trades:
                return None
            
            buy_volume = 0
            sell_volume = 0
            total_volume = 0
            
            for trade in trades:
                volume = float(trade['amount'])
                total_volume += volume
                
                if trade['side'] == 'buy':
                    buy_volume += volume
                else:
                    sell_volume += volume
            
            if total_volume == 0:
                return None
            
            buy_ratio = buy_volume / total_volume
            pressure = (buy_volume - sell_volume) / total_volume
            
            return {
                'buy_ratio': float(buy_ratio),
                'sell_ratio': float(1 - buy_ratio),
                'pressure': float(pressure),
                'total_volume': float(total_volume)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trades for {exchange_id} {symbol}: {e}")
            return None
    
    async def detect_volume_spike(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """Detect unusual volume spikes compared to average"""
        try:
            exchange = self.exchanges[exchange_id]
            
            ohlcv = await exchange.fetch_ohlcv(symbol, '1m', limit=60)
            if len(ohlcv) < 30:
                return None
            
            volumes = [candle[5] for candle in ohlcv]
            current_volume = float(volumes[-1])
            avg_volume = float(sum(volumes[:-1]) / len(volumes[:-1]))
            
            if avg_volume == 0:
                return None
            
            spike_ratio = current_volume / avg_volume
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'spike_ratio': float(spike_ratio),
                'is_spike': spike_ratio > 3.0
            }
            
        except Exception as e:
            logger.error(f"Error detecting volume spike for {exchange_id} {symbol}: {e}")
            return None
    
    async def analyze_exchange_flow(self, symbol: str) -> Optional[Dict]:
        """Analyze flow across all exchanges for a symbol"""
        try:
            tasks = []
            for exchange_id in self.exchanges.keys():
                if exchange_id not in self.failed_exchanges:
                    symbols = await self.load_exchange_symbols(exchange_id)
                    if symbol in symbols:
                        tasks.append(self.get_exchange_metrics(exchange_id, symbol))
            
            if not tasks:
                return None
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            valid_results = [r for r in results if isinstance(r, dict) and r is not None]
            if not valid_results:
                return None
            
            avg_imbalance = sum([r['imbalance'] for r in valid_results if r.get('imbalance')]) / len(valid_results)
            avg_pressure = sum([r['trade_pressure'] for r in valid_results if r.get('trade_pressure')]) / len(valid_results)
            
            total_volume = sum([r.get('volume', 0) for r in valid_results])
            spike_count = sum([1 for r in valid_results if r.get('has_spike', False)])
            
            flow_signal = 'NEUTRAL'
            confidence = 0
            
            if avg_imbalance > 0.3 and avg_pressure > 0.3:
                flow_signal = 'HEAVY_BUYING'
                confidence = min(abs(avg_imbalance) + abs(avg_pressure), 1.0) * 100
            elif avg_imbalance < -0.3 and avg_pressure < -0.3:
                flow_signal = 'HEAVY_SELLING'
                confidence = min(abs(avg_imbalance) + abs(avg_pressure), 1.0) * 100
            elif spike_count >= 2:
                if avg_pressure > 0:
                    flow_signal = 'VOLUME_SPIKE_BUY'
                else:
                    flow_signal = 'VOLUME_SPIKE_SELL'
                confidence = (spike_count / len(valid_results)) * 100
            
            return {
                'symbol': symbol,
                'flow_signal': flow_signal,
                'confidence': float(confidence),
                'avg_imbalance': float(avg_imbalance),
                'avg_pressure': float(avg_pressure),
                'total_volume': float(total_volume),
                'exchanges_analyzed': len(valid_results),
                'spike_count': spike_count,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing exchange flow for {symbol}: {e}")
            return None
    
    async def get_exchange_metrics(self, exchange_id: str, symbol: str) -> Optional[Dict]:
        """Get all metrics for a single exchange"""
        try:
            imbalance_task = self.calculate_order_book_imbalance(exchange_id, symbol)
            trades_task = self.get_recent_trades_pressure(exchange_id, symbol)
            spike_task = self.detect_volume_spike(exchange_id, symbol)
            
            imbalance, trades, spike = await asyncio.gather(
                imbalance_task, trades_task, spike_task, return_exceptions=True
            )
            
            return {
                'exchange': exchange_id,
                'imbalance': imbalance if not isinstance(imbalance, Exception) else 0,
                'trade_pressure': trades.get('pressure', 0) if isinstance(trades, dict) else 0,
                'volume': spike.get('current_volume', 0) if isinstance(spike, dict) else 0,
                'has_spike': spike.get('is_spike', False) if isinstance(spike, dict) else False
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics for {exchange_id} {symbol}: {e}")
            return None
    
    async def scan_all_symbols(self) -> List[Dict]:
        """Scan all symbols across all exchanges"""
        logger.info(f"Scanning spot markets across {len(self.exchanges)} exchanges...")
        
        tasks = []
        for symbol in self.symbols:
            tasks.append(self.analyze_exchange_flow(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, dict) and r is not None]
        
        significant_flows = [
            r for r in valid_results 
            if r.get('flow_signal') != 'NEUTRAL' and r.get('confidence', 0) >= 40
        ]
        
        logger.info(f"Found {len(significant_flows)} significant flow signals")
        
        return significant_flows
    
    async def save_spot_activity(self, flow_data: Dict):
        """Save spot activity to database"""
        try:
            db = next(get_db())
            
            spot_activity = SpotActivity(
                symbol=flow_data['symbol'],
                flow_signal=flow_data['flow_signal'],
                confidence=flow_data['confidence'],
                avg_imbalance=flow_data['avg_imbalance'],
                avg_pressure=flow_data['avg_pressure'],
                total_volume=flow_data['total_volume'],
                exchanges_count=flow_data['exchanges_analyzed'],
                spike_count=flow_data['spike_count'],
                created_at=datetime.utcnow()
            )
            
            db.add(spot_activity)
            db.commit()
            
        except Exception as e:
            logger.error(f"Error saving spot activity: {e}")
        finally:
            db.close()
    
    async def close(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            await exchange.close()

spot_monitor = SpotMarketMonitor()
