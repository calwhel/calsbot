"""
Professional Chart Generator for Twitter Posts
Generates clean, modern chart images for crypto coins
"""

import io
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import ccxt.async_support as ccxt

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed - chart generation disabled")


async def get_ohlcv_data(symbol: str, timeframe: str = '1h', limit: int = 48) -> Optional[List]:
    """Fetch OHLCV data from Binance (tries Futures first, then Spot)"""
    # Try Binance Futures first (more coins available)
    try:
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        await exchange.close()
        logger.info(f"✅ Got OHLCV from Binance Futures for {symbol}")
        return ohlcv
    except Exception as e:
        logger.warning(f"Futures OHLCV failed for {symbol}: {e}, trying spot...")
    
    # Fallback to Binance Spot
    try:
        exchange = ccxt.binance({'enableRateLimit': True})
        ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe, limit=limit)
        await exchange.close()
        logger.info(f"✅ Got OHLCV from Binance Spot for {symbol}")
        return ohlcv
    except Exception as e:
        logger.error(f"Failed to fetch OHLCV for {symbol} (both futures and spot): {e}")
        return None


async def generate_coin_chart(symbol: str, change_24h: float, current_price: float) -> Optional[bytes]:
    """Generate a professional high-quality chart image for a coin"""
    if not HAS_MATPLOTLIB:
        return None
    
    try:
        ohlcv = await get_ohlcv_data(symbol, '1h', 72)  # 72 hours for better context
        if not ohlcv or len(ohlcv) < 10:
            return None
        
        timestamps = [datetime.fromtimestamp(candle[0] / 1000) for candle in ohlcv]
        opens = [candle[1] for candle in ohlcv]
        closes = [candle[4] for candle in ohlcv]
        highs = [candle[2] for candle in ohlcv]
        lows = [candle[3] for candle in ohlcv]
        volumes = [candle[5] for candle in ohlcv]
        
        is_bullish = change_24h >= 0
        main_color = '#00D67D' if is_bullish else '#FF4757'
        secondary_color = '#00B86B' if is_bullish else '#E63946'
        bg_color = '#0A0E14'
        text_color = '#E6EDF3'
        grid_color = '#1C2128'
        accent_color = '#58A6FF'
        
        # Create figure with 2 subplots (price + volume)
        fig = plt.figure(figsize=(14, 8), facecolor=bg_color)
        gs = fig.add_gridspec(4, 1, hspace=0.05)
        ax_price = fig.add_subplot(gs[:3, 0])
        ax_vol = fig.add_subplot(gs[3, 0], sharex=ax_price)
        
        ax_price.set_facecolor(bg_color)
        ax_vol.set_facecolor(bg_color)
        
        # Calculate EMA for smoother line
        ema_period = 12
        ema = []
        multiplier = 2 / (ema_period + 1)
        for i, close in enumerate(closes):
            if i == 0:
                ema.append(close)
            else:
                ema.append((close - ema[-1]) * multiplier + ema[-1])
        
        # Main price line with gradient fill
        ax_price.fill_between(timestamps, closes, min(lows) * 0.998, 
                             color=main_color, alpha=0.08)
        ax_price.fill_between(timestamps, ema, min(lows) * 0.998, 
                             color=main_color, alpha=0.05)
        
        # Plot EMA as thinner line
        ax_price.plot(timestamps, ema, color=accent_color, linewidth=1.5, alpha=0.6, linestyle='--')
        
        # Main price line - thicker and smoother
        ax_price.plot(timestamps, closes, color=main_color, linewidth=3, solid_capstyle='round')
        
        # Highlight current price with glow effect
        ax_price.scatter([timestamps[-1]], [closes[-1]], color=main_color, s=150, zorder=5, alpha=0.3)
        ax_price.scatter([timestamps[-1]], [closes[-1]], color=main_color, s=80, zorder=6)
        ax_price.scatter([timestamps[-1]], [closes[-1]], color='white', s=20, zorder=7)
        
        # Add horizontal line at current price
        ax_price.axhline(y=closes[-1], color=main_color, linestyle=':', linewidth=1, alpha=0.5)
        
        # Price annotations
        price_range = max(highs) - min(lows)
        ax_price.set_ylim(min(lows) - price_range * 0.08, max(highs) + price_range * 0.20)
        
        # Volume bars
        vol_colors = [main_color if closes[i] >= opens[i] else secondary_color for i in range(len(volumes))]
        ax_vol.bar(timestamps, volumes, color=vol_colors, alpha=0.6, width=0.03)
        
        # Formatting
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        
        for ax in [ax_price, ax_vol]:
            ax.tick_params(colors=text_color, labelsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color(grid_color)
            ax.spines['left'].set_color(grid_color)
            ax.grid(True, alpha=0.2, color=grid_color, linestyle='-', linewidth=0.5)
        
        plt.setp(ax_price.get_xticklabels(), visible=False)
        ax_vol.set_ylabel('Volume', color=text_color, fontsize=10)
        
        # Title with symbol
        sign = '+' if change_24h >= 0 else ''
        price_str = f"${current_price:,.6f}" if current_price < 0.01 else f"${current_price:,.4f}" if current_price < 1 else f"${current_price:,.2f}"
        title = f"${symbol}"
        ax_price.set_title(title, fontsize=28, fontweight='bold', color=text_color, pad=25, loc='left', fontfamily='monospace')
        
        # Price and change on right
        ax_price.text(0.99, 0.97, price_str, transform=ax_price.transAxes, fontsize=22, 
                     fontweight='bold', color=text_color, ha='right', va='top', fontfamily='monospace')
        
        change_text = f"{sign}{change_24h:.2f}%"
        ax_price.text(0.99, 0.87, change_text, transform=ax_price.transAxes, fontsize=18, 
                     fontweight='bold', color=main_color, ha='right', va='top')
        
        # 24h High/Low
        ax_price.text(0.01, 0.97, f"H: ${max(highs[-24:]):,.4f}" if max(highs[-24:]) < 1 else f"H: ${max(highs[-24:]):,.2f}", 
                     transform=ax_price.transAxes, fontsize=10, color='#8B949E', ha='left', va='top')
        ax_price.text(0.01, 0.91, f"L: ${min(lows[-24:]):,.4f}" if min(lows[-24:]) < 1 else f"L: ${min(lows[-24:]):,.2f}", 
                     transform=ax_price.transAxes, fontsize=10, color='#8B949E', ha='left', va='top')
        
        # Branding
        ax_vol.text(0.01, -0.35, "TradeHub AI", transform=ax_vol.transAxes, fontsize=11,
                   color='#484F58', ha='left', va='top', fontweight='bold')
        ax_vol.text(0.99, -0.35, "72H Chart • 1H Candles", transform=ax_vol.transAxes, fontsize=10,
                   color='#484F58', ha='right', va='top')
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=200, facecolor=bg_color, 
                   edgecolor='none', bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate chart for {symbol}: {e}")
        return None


async def generate_market_overview_chart(coins: List[Dict]) -> Optional[bytes]:
    """Generate a market overview bar chart"""
    if not HAS_MATPLOTLIB or not coins:
        return None
    
    try:
        bg_color = '#0D1117'
        text_color = '#FFFFFF'
        
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        
        symbols = [c['symbol'] for c in coins[:8]]
        changes = [c['change'] for c in coins[:8]]
        colors = ['#00C853' if c >= 0 else '#FF1744' for c in changes]
        
        bars = ax.barh(symbols, changes, color=colors, height=0.6, edgecolor='none')
        
        for bar, change in zip(bars, changes):
            width = bar.get_width()
            sign = '+' if change >= 0 else ''
            ax.text(width + (0.5 if change >= 0 else -0.5), bar.get_y() + bar.get_height()/2,
                   f'{sign}{change:.1f}%', ha='left' if change >= 0 else 'right',
                   va='center', color=text_color, fontsize=11, fontweight='bold')
        
        ax.axvline(x=0, color='#586069', linewidth=1)
        ax.set_xlabel('24h Change %', color=text_color, fontsize=12)
        ax.tick_params(colors=text_color, labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#21262D')
        ax.spines['left'].set_color('#21262D')
        
        ax.set_title('TOP MOVERS', fontsize=18, fontweight='bold', color=text_color, pad=15)
        ax.text(0.98, 0.02, "TradeHub AI", transform=ax.transAxes, fontsize=10,
               color='#586069', ha='right', va='bottom', alpha=0.7)
        
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, facecolor=bg_color,
                   edgecolor='none', bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        buf.seek(0)
        
        return buf.getvalue()
        
    except Exception as e:
        logger.error(f"Failed to generate market overview chart: {e}")
        return None
