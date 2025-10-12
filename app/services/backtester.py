import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class Backtester:
    """Backtest trading strategies on historical data"""
    
    def __init__(self, exchange_name: str = 'kucoin'):
        self.exchange = getattr(ccxt, exchange_name)()
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1h', days: int = 90) -> Optional[pd.DataFrame]:
        """Fetch historical OHLCV data"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EMA and other indicators"""
        # EMA indicators
        df['ema_fast'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Volume
        df['volume_avg'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Generate buy/sell signals from historical data"""
        signals = []
        
        for i in range(50, len(df)):  # Start after indicators are stable
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # Skip if NaN values
            if pd.isna([current['ema_fast'], current['ema_slow'], current['rsi'], current['atr']]).any():
                continue
            
            # Volume confirmation
            if current['volume'] < current['volume_avg']:
                continue
            
            # LONG signal
            if (prev['ema_fast'] <= prev['ema_slow'] and 
                current['ema_fast'] > current['ema_slow'] and 
                current['ema_fast'] > current['ema_trend'] and
                current['rsi'] < 70):
                
                entry_price = current['close']
                stop_loss = entry_price - (2 * current['atr'])
                take_profit = entry_price + (3 * current['atr'])
                
                signals.append({
                    'timestamp': current.name,
                    'direction': 'LONG',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': current['atr']
                })
            
            # SHORT signal
            elif (prev['ema_fast'] >= prev['ema_slow'] and 
                  current['ema_fast'] < current['ema_slow'] and 
                  current['ema_fast'] < current['ema_trend'] and
                  current['rsi'] > 30):
                
                entry_price = current['close']
                stop_loss = entry_price + (2 * current['atr'])
                take_profit = entry_price - (3 * current['atr'])
                
                signals.append({
                    'timestamp': current.name,
                    'direction': 'SHORT',
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'atr': current['atr']
                })
        
        return signals
    
    def simulate_trades(self, df: pd.DataFrame, signals: List[Dict]) -> List[Dict]:
        """Simulate trade execution and calculate results"""
        trades = []
        
        for signal in signals:
            entry_time = signal['timestamp']
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            direction = signal['direction']
            
            # Find exit point
            future_data = df[df.index > entry_time].head(100)  # Look ahead max 100 bars
            
            exit_price = None
            exit_time = None
            exit_reason = None
            
            for timestamp, row in future_data.iterrows():
                if direction == 'LONG':
                    if row['high'] >= take_profit:
                        exit_price = take_profit
                        exit_time = timestamp
                        exit_reason = 'TP'
                        break
                    elif row['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_time = timestamp
                        exit_reason = 'SL'
                        break
                else:  # SHORT
                    if row['low'] <= take_profit:
                        exit_price = take_profit
                        exit_time = timestamp
                        exit_reason = 'TP'
                        break
                    elif row['high'] >= stop_loss:
                        exit_price = stop_loss
                        exit_time = timestamp
                        exit_reason = 'SL'
                        break
            
            # If no exit found, close at last price
            if exit_price is None and len(future_data) > 0:
                exit_price = future_data.iloc[-1]['close']
                exit_time = future_data.index[-1]
                exit_reason = 'EOD'
            
            if exit_price:
                # Calculate PnL
                if direction == 'LONG':
                    pnl_percent = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_percent = ((entry_price - exit_price) / entry_price) * 100
                
                # With 10x leverage
                pnl_percent_10x = pnl_percent * 10
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_percent': pnl_percent,
                    'pnl_percent_10x': pnl_percent_10x,
                    'exit_reason': exit_reason
                })
        
        return trades
    
    def calculate_metrics(self, trades: List[Dict]) -> Dict:
        """Calculate backtest performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'total_return': 0
            }
        
        winning_trades = [t for t in trades if t['pnl_percent_10x'] > 0]
        losing_trades = [t for t in trades if t['pnl_percent_10x'] < 0]
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        avg_win = np.mean([t['pnl_percent_10x'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_percent_10x'] for t in losing_trades]) if losing_trades else 0
        
        total_profit = sum([t['pnl_percent_10x'] for t in winning_trades])
        total_loss = abs(sum([t['pnl_percent_10x'] for t in losing_trades]))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Calculate drawdown
        cumulative_returns = np.cumsum([t['pnl_percent_10x'] for t in trades])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = running_max - cumulative_returns
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        total_return = sum([t['pnl_percent_10x'] for t in trades])
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'best_trade': max(trades, key=lambda x: x['pnl_percent_10x']) if trades else None,
            'worst_trade': min(trades, key=lambda x: x['pnl_percent_10x']) if trades else None
        }
    
    def run_backtest(self, symbol: str, timeframe: str = '1h', days: int = 90) -> Dict:
        """Run complete backtest for a symbol"""
        logger.info(f"Running backtest for {symbol} on {timeframe} for {days} days")
        
        # Fetch data
        df = self.fetch_historical_data(symbol, timeframe, days)
        if df is None or len(df) < 50:
            return {'error': 'Insufficient data'}
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Generate signals
        signals = self.generate_signals(df)
        
        if not signals:
            return {'error': 'No signals generated'}
        
        # Simulate trades
        trades = self.simulate_trades(df, signals)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        metrics['symbol'] = symbol
        metrics['timeframe'] = timeframe
        metrics['period_days'] = days
        metrics['signals_generated'] = len(signals)
        
        return metrics
