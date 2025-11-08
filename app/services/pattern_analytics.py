"""
Pattern Performance Analytics
Tracks win rate, average PnL, and trade count per signal pattern type
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models import Signal, Trade

logger = logging.getLogger(__name__)


def calculate_pattern_performance(db: Session, days: int = 30) -> List[Dict]:
    """
    Calculate performance metrics for each pattern type (live trades only)
    
    Args:
        db: Database session
        days: Number of days to look back (default 30)
    
    Returns:
        List of dicts with pattern performance metrics
    """
    start_date = datetime.utcnow() - timedelta(days=days)
    
    # Get all signals with patterns from the time period
    # Filter out None and empty strings
    signals = db.query(Signal).filter(
        Signal.created_at >= start_date,
        Signal.pattern.isnot(None),
        Signal.pattern != ''
    ).all()
    
    if not signals:
        return []
    
    # Group by pattern
    pattern_stats = {}
    
    for signal in signals:
        pattern_name = signal.pattern
        
        if pattern_name not in pattern_stats:
            pattern_stats[pattern_name] = {
                'pattern': pattern_name,
                'signal_type': signal.signal_type,
                'total_signals': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'total_pnl_percent': 0.0,
                'avg_confidence': 0.0,
                'confidences': []
            }
        
        stats = pattern_stats[pattern_name]
        stats['total_signals'] += 1
        
        if signal.confidence:
            stats['confidences'].append(signal.confidence)
        
        # Get live trades for this signal
        all_trades = db.query(Trade).filter(
            Trade.signal_id == signal.id,
            Trade.status.in_(['closed', 'stopped', 'tp_hit', 'sl_hit'])
        ).all()
        
        for trade in all_trades:
            stats['total_trades'] += 1
            
            pnl = trade.pnl or 0
            pnl_percent = trade.pnl_percent or 0
            
            stats['total_pnl'] += pnl
            stats['total_pnl_percent'] += pnl_percent
            
            if pnl > 0:
                stats['winning_trades'] += 1
            elif pnl < 0:
                stats['losing_trades'] += 1
    
    # Calculate final metrics
    results = []
    for pattern_name, stats in pattern_stats.items():
        # Calculate averages
        if stats['total_trades'] > 0:
            win_rate = (stats['winning_trades'] / stats['total_trades']) * 100
            avg_pnl = stats['total_pnl'] / stats['total_trades']
            avg_pnl_percent = stats['total_pnl_percent'] / stats['total_trades']
        else:
            win_rate = 0
            avg_pnl = 0
            avg_pnl_percent = 0
        
        if stats['confidences']:
            avg_confidence = sum(stats['confidences']) / len(stats['confidences'])
        else:
            avg_confidence = 0
        
        results.append({
            'pattern': pattern_name,
            'signal_type': stats['signal_type'],
            'total_signals': stats['total_signals'],
            'total_trades': stats['total_trades'],
            'win_rate': round(win_rate, 1),
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'total_pnl': round(stats['total_pnl'], 2),
            'avg_pnl': round(avg_pnl, 2),
            'avg_pnl_percent': round(avg_pnl_percent, 2),
            'avg_confidence': round(avg_confidence, 1)
        })
    
    # Sort by win rate (highest first)
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    return results


def get_top_patterns(db: Session, days: int = 30, limit: int = 5, include_paper: bool = True) -> List[Dict]:
    """Get top performing patterns by win rate"""
    all_patterns = calculate_pattern_performance(db, days, include_paper=include_paper)
    
    # Filter patterns with at least 3 trades
    valid_patterns = [p for p in all_patterns if p['total_trades'] >= 3]
    
    return valid_patterns[:limit]


def get_worst_patterns(db: Session, days: int = 30, limit: int = 5, include_paper: bool = True) -> List[Dict]:
    """Get worst performing patterns by win rate"""
    all_patterns = calculate_pattern_performance(db, days, include_paper=include_paper)
    
    # Filter patterns with at least 3 trades
    valid_patterns = [p for p in all_patterns if p['total_trades'] >= 3]
    
    # Sort by win rate (lowest first)
    valid_patterns.sort(key=lambda x: x['win_rate'])
    
    return valid_patterns[:limit]


def format_pattern_performance_message(patterns: List[Dict], title: str = "Pattern Performance") -> str:
    """Format pattern performance data into readable message"""
    if not patterns:
        return f"<b>{title}</b>\n━━━━━━━━━━━━━━━━━━━━\n\nNo pattern data available yet.\nTrade more to see analytics!"
    
    msg = f"<b>{title}</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
    
    for i, p in enumerate(patterns, 1):
        # Emoji based on win rate
        if p['win_rate'] >= 70:
            emoji = "🟢"
        elif p['win_rate'] >= 50:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        # Emoji for PnL
        pnl_emoji = "📈" if p['avg_pnl'] > 0 else "📉"
        
        msg += f"{i}. {emoji} <b>{p['pattern']}</b>\n"
        msg += f"   Win Rate: <b>{p['win_rate']}%</b> ({p['winning_trades']}W / {p['losing_trades']}L)\n"
        msg += f"   {pnl_emoji} Avg PnL: <b>{p['avg_pnl_percent']:+.2f}%</b> (${p['avg_pnl']:+.2f})\n"
        msg += f"   📊 Trades: {p['total_trades']} | Signals: {p['total_signals']}\n"
        msg += f"   💡 Avg Confidence: {p['avg_confidence']:.0f}%\n"
        msg += "━━━━━━━━━━━━━━━━━━━━\n"
    
    return msg
