"""
Social Trade Logger - Comprehensive logging for Social & News trades
Tracks every trade from signal to close with full win/loss analysis
"""
import logging
from datetime import datetime
from typing import Dict, Optional
from sqlalchemy.orm import Session

from app.models import SocialTradeLog, Trade, User, UserPreferences

logger = logging.getLogger(__name__)


async def log_social_signal_generated(
    db: Session,
    user_id: int,
    signal: Dict,
    risk_level: str = None
) -> Optional[int]:
    """Log when a social/news signal is generated for a user"""
    try:
        signal_type = signal.get('trade_type', 'SOCIAL_SIGNAL')
        
        log_entry = SocialTradeLog(
            user_id=user_id,
            symbol=signal.get('symbol', ''),
            direction=signal.get('direction', ''),
            signal_type=signal_type,
            strategy=signal.get('strategy', ''),
            galaxy_score=signal.get('galaxy_score'),
            alt_rank=signal.get('alt_rank'),
            sentiment_score=signal.get('sentiment'),
            news_title=signal.get('news_title'),
            news_trigger=signal.get('trigger_reason'),
            signal_score=signal.get('confidence'),
            reasoning=signal.get('reasoning'),
            rsi_at_entry=signal.get('rsi'),
            price_change_24h=signal.get('change_24h'),
            entry_price=signal.get('entry_price'),
            stop_loss=signal.get('stop_loss'),
            take_profit=signal.get('take_profit') or signal.get('take_profit_1'),
            tp_percent=signal.get('tp_percent'),
            sl_percent=signal.get('sl_percent'),
            status='pending',
            signal_time=datetime.utcnow(),
            user_risk_level=risk_level
        )
        
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        
        logger.info(f"📊 SOCIAL LOG #{log_entry.id}: Signal generated")
        logger.info(f"   → {signal.get('symbol')} {signal.get('direction')} | Type: {signal_type}")
        logger.info(f"   → Score: {signal.get('confidence')} | Risk: {risk_level}")
        
        return log_entry.id
        
    except Exception as e:
        logger.error(f"Error logging social signal: {e}")
        return None


async def log_social_trade_opened(
    db: Session,
    log_id: int,
    trade: Trade,
    position_size: float,
    leverage: int
) -> bool:
    """Update log when trade is actually opened on exchange"""
    try:
        log_entry = db.query(SocialTradeLog).filter(SocialTradeLog.id == log_id).first()
        if not log_entry:
            return False
        
        log_entry.trade_id = trade.id
        log_entry.status = 'open'
        log_entry.open_time = datetime.utcnow()
        log_entry.position_size = position_size
        log_entry.leverage = leverage
        log_entry.entry_price = trade.entry_price
        
        db.commit()
        
        logger.info(f"📊 SOCIAL LOG #{log_id}: Trade OPENED")
        logger.info(f"   → Trade ID: {trade.id} | Size: ${position_size:.2f} | {leverage}x")
        logger.info(f"   → Entry: ${trade.entry_price:.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating social trade log: {e}")
        return False


async def log_social_trade_closed(
    db: Session,
    trade: Trade
) -> bool:
    """Update log when trade is closed with final result"""
    try:
        if trade.trade_type not in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
            return False
        
        log_entry = db.query(SocialTradeLog).filter(
            SocialTradeLog.trade_id == trade.id
        ).first()
        
        if not log_entry:
            log_entry = db.query(SocialTradeLog).filter(
                SocialTradeLog.user_id == trade.user_id,
                SocialTradeLog.symbol == trade.symbol,
                SocialTradeLog.status == 'open'
            ).order_by(SocialTradeLog.signal_time.desc()).first()
        
        if not log_entry:
            logger.warning(f"No social log found for trade {trade.id}")
            return False
        
        pnl = trade.pnl or 0
        pnl_percent = trade.pnl_percent or 0
        
        if pnl > 0:
            result = 'WIN'
        elif pnl < 0:
            result = 'LOSS'
        else:
            result = 'BREAKEVEN'
        
        # Calculate ROI with leverage
        leverage = log_entry.leverage or 1
        position_size = log_entry.position_size or trade.position_size or 1
        roi_percent = (pnl / position_size) * 100 if position_size > 0 else 0
        
        log_entry.status = trade.status
        log_entry.result = result
        log_entry.exit_price = trade.exit_price
        log_entry.pnl = pnl
        log_entry.pnl_percent = pnl_percent
        log_entry.roi_percent = roi_percent
        log_entry.close_time = datetime.utcnow()
        
        if log_entry.open_time:
            duration = (datetime.utcnow() - log_entry.open_time).total_seconds() / 60
            log_entry.duration_minutes = int(duration)
        
        db.commit()
        
        result_emoji = "✅" if result == 'WIN' else ("❌" if result == 'LOSS' else "➖")
        pnl_sign = "+" if pnl >= 0 else ""
        roi_sign = "+" if roi_percent >= 0 else ""
        
        logger.info(f"📊 SOCIAL LOG #{log_entry.id}: Trade CLOSED - {result_emoji} {result}")
        logger.info(f"   → {log_entry.symbol} {log_entry.direction} | {log_entry.signal_type}")
        logger.info(f"   → Entry: ${log_entry.entry_price:.6f} → Exit: ${trade.exit_price:.6f}")
        logger.info(f"   → PnL: {pnl_sign}${pnl:.2f} | ROI: {roi_sign}{roi_percent:.2f}% @ {leverage}x")
        logger.info(f"   → Duration: {log_entry.duration_minutes} min | Status: {trade.status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error closing social trade log: {e}")
        return False


def get_social_trade_stats(db: Session, user_id: int = None, days: int = 7) -> Dict:
    """Get win/loss statistics for social trades"""
    try:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(SocialTradeLog).filter(
            SocialTradeLog.signal_time >= cutoff,
            SocialTradeLog.result.isnot(None)
        )
        
        if user_id:
            query = query.filter(SocialTradeLog.user_id == user_id)
        
        logs = query.all()
        
        total = len(logs)
        wins = sum(1 for l in logs if l.result == 'WIN')
        losses = sum(1 for l in logs if l.result == 'LOSS')
        breakeven = sum(1 for l in logs if l.result == 'BREAKEVEN')
        
        total_pnl = sum(l.pnl or 0 for l in logs)
        avg_win = sum(l.pnl for l in logs if l.result == 'WIN' and l.pnl) / wins if wins > 0 else 0
        avg_loss = sum(l.pnl for l in logs if l.result == 'LOSS' and l.pnl) / losses if losses > 0 else 0
        
        by_type = {}
        for signal_type in ['SOCIAL_SIGNAL', 'SOCIAL_SHORT', 'NEWS_SIGNAL']:
            type_logs = [l for l in logs if l.signal_type == signal_type]
            type_wins = sum(1 for l in type_logs if l.result == 'WIN')
            by_type[signal_type] = {
                'total': len(type_logs),
                'wins': type_wins,
                'win_rate': (type_wins / len(type_logs) * 100) if type_logs else 0,
                'pnl': sum(l.pnl or 0 for l in type_logs)
            }
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'breakeven': breakeven,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'by_type': by_type,
            'days': days
        }
        
    except Exception as e:
        logger.error(f"Error getting social trade stats: {e}")
        return {}


def get_recent_social_trades(db: Session, user_id: int = None, limit: int = 20) -> list:
    """Get recent social trades with full details"""
    try:
        query = db.query(SocialTradeLog).filter(
            SocialTradeLog.result.isnot(None)
        ).order_by(SocialTradeLog.close_time.desc())
        
        if user_id:
            query = query.filter(SocialTradeLog.user_id == user_id)
        
        return query.limit(limit).all()
        
    except Exception as e:
        logger.error(f"Error getting recent social trades: {e}")
        return []
