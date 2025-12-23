from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from app.models import Signal, Trade, User
import logging

logger = logging.getLogger(__name__)


class AnalyticsService:
    
    @staticmethod
    def update_signal_outcome(db: Session, signal_id: int):
        signal = db.query(Signal).filter(Signal.id == signal_id).first()
        if not signal:
            return
        
        trades = db.query(Trade).filter(
            Trade.signal_id == signal_id,
            Trade.status == "closed"
        ).all()
        
        if not trades:
            return
        
        total_pnl = sum(t.pnl or 0 for t in trades)
        avg_pnl_percent = sum(t.pnl_percent or 0 for t in trades) / len(trades) if trades else 0
        
        tp1_hits = sum(1 for t in trades if t.tp1_hit)
        tp2_hits = sum(1 for t in trades if t.tp2_hit)
        
        winning_trades = sum(1 for t in trades if (t.pnl_percent or 0) > 0)
        losing_trades = sum(1 for t in trades if (t.pnl_percent or 0) < 0)
        
        if tp1_hits > 0 or tp2_hits > 0:
            outcome = "won"
        elif winning_trades > losing_trades:
            outcome = "won"
        elif losing_trades > winning_trades:
            outcome = "lost"
        elif avg_pnl_percent >= 0:
            outcome = "won"
        else:
            outcome = "lost"
        
        signal.outcome = outcome
        signal.total_pnl = float(total_pnl)
        signal.total_pnl_percent = float(avg_pnl_percent)
        signal.trades_count = len(trades)
        db.commit()
        
        logger.info(f"Updated signal {signal_id} outcome: {outcome} | PnL: {avg_pnl_percent:.1f}% | TP1 hits: {tp1_hits} | TP2 hits: {tp2_hits} | W/L: {winning_trades}/{losing_trades}")
    
    @staticmethod
    def recalculate_all_signal_outcomes(db: Session, days: int = 30) -> Dict:
        """Recalculate outcomes for all signals with closed trades"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        signals = db.query(Signal).filter(
            Signal.created_at >= cutoff
        ).all()
        
        updated = 0
        for signal in signals:
            try:
                AnalyticsService.update_signal_outcome(db, signal.id)
                updated += 1
            except Exception as e:
                logger.error(f"Error updating signal {signal.id}: {e}")
        
        logger.info(f"Recalculated outcomes for {updated} signals")
        return {"updated": updated, "total": len(signals)}
    
    @staticmethod
    def get_performance_stats(db: Session, days: int = 30) -> Dict:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        signals = db.query(Signal).filter(
            Signal.created_at >= cutoff,
            Signal.outcome.isnot(None)
        ).all()
        
        if not signals:
            return {
                "total_signals": 0,
                "won": 0,
                "lost": 0,
                "breakeven": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "best_signal": None,
                "worst_signal": None
            }
        
        won = [s for s in signals if s.outcome == "won"]
        lost = [s for s in signals if s.outcome == "lost"]
        breakeven = [s for s in signals if s.outcome == "breakeven"]
        
        total_signals = len(signals)
        win_rate = (len(won) / total_signals * 100) if total_signals > 0 else 0
        avg_pnl = sum(s.total_pnl_percent for s in signals) / total_signals if total_signals > 0 else 0
        total_pnl = sum(s.total_pnl for s in signals)
        
        best_signal = max(signals, key=lambda s: s.total_pnl_percent) if signals else None
        worst_signal = min(signals, key=lambda s: s.total_pnl_percent) if signals else None
        
        return {
            "total_signals": total_signals,
            "won": len(won),
            "lost": len(lost),
            "breakeven": len(breakeven),
            "win_rate": float(win_rate),
            "avg_pnl": float(avg_pnl),
            "total_pnl": float(total_pnl),
            "best_signal": {
                "symbol": best_signal.symbol,
                "direction": best_signal.direction,
                "pnl": best_signal.total_pnl_percent,
                "type": best_signal.signal_type
            } if best_signal else None,
            "worst_signal": {
                "symbol": worst_signal.symbol,
                "direction": worst_signal.direction,
                "pnl": worst_signal.total_pnl_percent,
                "type": worst_signal.signal_type
            } if worst_signal else None
        }
    
    @staticmethod
    def get_symbol_performance(db: Session, days: int = 30) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = db.query(
            Signal.symbol,
            func.count(Signal.id).label('count'),
            func.avg(Signal.total_pnl_percent).label('avg_pnl'),
            func.sum(Signal.total_pnl).label('total_pnl')
        ).filter(
            Signal.created_at >= cutoff,
            Signal.outcome.isnot(None)
        ).group_by(Signal.symbol).order_by(func.avg(Signal.total_pnl_percent).desc()).all()
        
        return [
            {
                "symbol": r.symbol,
                "count": r.count,
                "avg_pnl": float(r.avg_pnl or 0),
                "total_pnl": float(r.total_pnl or 0)
            }
            for r in results
        ]
    
    @staticmethod
    def get_signal_type_performance(db: Session, days: int = 30) -> Dict:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        technical = db.query(Signal).filter(
            Signal.created_at >= cutoff,
            Signal.signal_type == 'technical',
            Signal.outcome.isnot(None)
        ).all()
        
        news = db.query(Signal).filter(
            Signal.created_at >= cutoff,
            Signal.signal_type == 'news',
            Signal.outcome.isnot(None)
        ).all()
        
        def calc_stats(signals):
            if not signals:
                return {"count": 0, "win_rate": 0, "avg_pnl": 0}
            won = len([s for s in signals if s.outcome == "won"])
            win_rate = (won / len(signals) * 100) if signals else 0
            avg_pnl = sum(s.total_pnl_percent for s in signals) / len(signals) if signals else 0
            return {
                "count": len(signals),
                "win_rate": float(win_rate),
                "avg_pnl": float(avg_pnl)
            }
        
        return {
            "technical": calc_stats(technical),
            "news": calc_stats(news)
        }
    
    @staticmethod
    def get_timeframe_performance(db: Session, days: int = 30) -> List[Dict]:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        results = db.query(
            Signal.timeframe,
            func.count(Signal.id).label('count'),
            func.avg(Signal.total_pnl_percent).label('avg_pnl')
        ).filter(
            Signal.created_at >= cutoff,
            Signal.outcome.isnot(None)
        ).group_by(Signal.timeframe).order_by(func.avg(Signal.total_pnl_percent).desc()).all()
        
        return [
            {
                "timeframe": r.timeframe,
                "count": r.count,
                "avg_pnl": float(r.avg_pnl or 0)
            }
            for r in results
        ]
