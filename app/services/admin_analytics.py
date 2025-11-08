import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session
from app.models import User, Trade, Signal, UserPreference
from app.database import SessionLocal

logger = logging.getLogger(__name__)


class AdminAnalytics:
    """Comprehensive analytics for admin dashboard - track growth, performance, and system health"""
    
    @staticmethod
    def get_user_growth_metrics(db: Session, days: int = 30) -> Dict:
        """Calculate user growth metrics (DAU, WAU, MAU, new users, retention)"""
        try:
            now = datetime.utcnow()
            
            # Total users
            total_users = db.query(User).count()
            approved_users = db.query(User).filter(User.approved == True).count()
            pending_users = db.query(User).filter(User.approved == False, User.banned == False).count()
            banned_users = db.query(User).filter(User.banned == True).count()
            
            # New users in last 24h, 7d, 30d
            yesterday = now - timedelta(days=1)
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            new_users_today = db.query(User).filter(User.created_at >= yesterday).count()
            new_users_week = db.query(User).filter(User.created_at >= week_ago).count()
            new_users_month = db.query(User).filter(User.created_at >= month_ago).count()
            
            # Active users (users with live trades in period)
            total_dau = db.query(Trade.user_id).filter(Trade.opened_at >= yesterday).distinct().count()
            total_wau = db.query(Trade.user_id).filter(Trade.opened_at >= week_ago).distinct().count()
            mau = db.query(Trade.user_id).filter(Trade.opened_at >= month_ago).distinct().count()
            
            # Retention rate (users who traded this week vs last week)
            last_week_traders = db.query(Trade.user_id).filter(
                Trade.opened_at >= now - timedelta(days=14),
                Trade.opened_at < week_ago
            ).distinct().count()
            
            retention_rate = (total_wau / last_week_traders * 100) if last_week_traders > 0 else 0
            
            return {
                "total_users": total_users,
                "approved_users": approved_users,
                "pending_users": pending_users,
                "banned_users": banned_users,
                "new_users_today": new_users_today,
                "new_users_week": new_users_week,
                "new_users_month": new_users_month,
                "dau": total_dau,
                "wau": total_wau,
                "mau": mau,
                "retention_rate": retention_rate,
                "engagement_rate": (total_wau / approved_users * 100) if approved_users > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting user growth metrics: {e}")
            return {}
    
    @staticmethod
    def get_signal_performance_summary(db: Session, days: int = 30) -> Dict:
        """Analyze signal performance over time"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Total signals generated
            total_signals = db.query(Signal).filter(Signal.created_at >= cutoff).count()
            
            # Signals by type
            technical_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.signal_type == 'technical'
            ).count()
            
            news_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.signal_type == 'news'
            ).count()
            
            spot_flow_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.signal_type == 'spot_flow'
            ).count()
            
            # Signal outcomes
            won_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.outcome == 'won'
            ).count()
            
            lost_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.outcome == 'lost'
            ).count()
            
            breakeven_signals = db.query(Signal).filter(
                Signal.created_at >= cutoff,
                Signal.outcome == 'breakeven'
            ).count()
            
            total_tracked = won_signals + lost_signals + breakeven_signals
            win_rate = (won_signals / total_tracked * 100) if total_tracked > 0 else 0
            
            # Average PnL
            avg_pnl = db.query(func.avg(Signal.total_pnl_percent)).filter(
                Signal.created_at >= cutoff,
                Signal.outcome.isnot(None)
            ).scalar() or 0
            
            # Best performing symbol
            best_symbol = db.query(
                Signal.symbol,
                func.avg(Signal.total_pnl_percent).label('avg_pnl')
            ).filter(
                Signal.created_at >= cutoff,
                Signal.outcome.isnot(None)
            ).group_by(Signal.symbol).order_by(func.avg(Signal.total_pnl_percent).desc()).first()
            
            return {
                "total_signals": total_signals,
                "technical_signals": technical_signals,
                "news_signals": news_signals,
                "spot_flow_signals": spot_flow_signals,
                "won_signals": won_signals,
                "lost_signals": lost_signals,
                "breakeven_signals": breakeven_signals,
                "win_rate": win_rate,
                "avg_pnl_percent": float(avg_pnl) if avg_pnl else 0,
                "best_symbol": best_symbol[0] if best_symbol else "N/A",
                "best_symbol_pnl": float(best_symbol[1]) if best_symbol else 0
            }
        except Exception as e:
            logger.error(f"Error getting signal performance: {e}")
            return {}
    
    @staticmethod
    def get_exchange_usage_stats(db: Session) -> Dict:
        """Track which exchanges users are connecting to"""
        try:
            total_users_with_prefs = db.query(UserPreference).count()
            
            mexc_users = db.query(UserPreference).filter(
                UserPreference.mexc_api_key.isnot(None),
                UserPreference.mexc_api_secret.isnot(None)
            ).count()
            
            kucoin_users = db.query(UserPreference).filter(
                UserPreference.kucoin_api_key.isnot(None),
                UserPreference.kucoin_api_secret.isnot(None)
            ).count()
            
            okx_users = db.query(UserPreference).filter(
                UserPreference.okx_api_key.isnot(None),
                UserPreference.okx_api_secret.isnot(None)
            ).count()
            
            auto_trading_enabled = db.query(UserPreference).filter(
                UserPreference.auto_trading_enabled == True
            ).count()
            
            # Preferred exchange distribution
            mexc_preferred = db.query(UserPreference).filter(
                UserPreference.preferred_exchange == 'mexc'
            ).count()
            
            kucoin_preferred = db.query(UserPreference).filter(
                UserPreference.preferred_exchange == 'kucoin'
            ).count()
            
            okx_preferred = db.query(UserPreference).filter(
                UserPreference.preferred_exchange == 'okx'
            ).count()
            
            return {
                "total_configured_users": total_users_with_prefs,
                "mexc_users": mexc_users,
                "kucoin_users": kucoin_users,
                "okx_users": okx_users,
                "auto_trading_enabled": auto_trading_enabled,
                "mexc_preferred": mexc_preferred,
                "kucoin_preferred": kucoin_preferred,
                "okx_preferred": okx_preferred
            }
        except Exception as e:
            logger.error(f"Error getting exchange usage: {e}")
            return {}
    
    @staticmethod
    def get_trading_volume_stats(db: Session, days: int = 30) -> Dict:
        """Calculate trading volume and activity stats"""
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            # Live trades only
            total_live_trades = db.query(Trade).filter(Trade.opened_at >= cutoff).count()
            open_live_trades = db.query(Trade).filter(
                Trade.opened_at >= cutoff,
                Trade.status == 'open'
            ).count()
            closed_live_trades = db.query(Trade).filter(
                Trade.opened_at >= cutoff,
                Trade.status == 'closed'
            ).count()
            
            # PnL stats
            total_live_pnl = db.query(func.sum(Trade.pnl)).filter(
                Trade.opened_at >= cutoff,
                Trade.status == 'closed'
            ).scalar() or 0
            
            avg_trade_pnl = db.query(func.avg(Trade.pnl)).filter(
                Trade.opened_at >= cutoff,
                Trade.status == 'closed'
            ).scalar() or 0
            
            # Most active traders
            top_trader = db.query(
                Trade.user_id,
                func.count(Trade.id).label('trade_count')
            ).filter(
                Trade.opened_at >= cutoff
            ).group_by(Trade.user_id).order_by(func.count(Trade.id).desc()).first()
            
            return {
                "total_live_trades": total_live_trades,
                "open_live_trades": open_live_trades,
                "closed_live_trades": closed_live_trades,
                "total_live_pnl": float(total_live_pnl),
                "avg_trade_pnl": float(avg_trade_pnl) if avg_trade_pnl else 0,
                "most_active_user_id": top_trader[0] if top_trader else None,
                "most_active_user_trades": top_trader[1] if top_trader else 0
            }
        except Exception as e:
            logger.error(f"Error getting trading volume stats: {e}")
            return {}
    
    @staticmethod
    def get_system_health_metrics(db: Session) -> Dict:
        """Monitor system health and potential issues"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            # Recent signal generation rate
            signals_last_hour = db.query(Signal).filter(Signal.created_at >= hour_ago).count()
            
            # Recent trade execution rate (live trades only)
            trades_last_hour = db.query(Trade).filter(Trade.opened_at >= hour_ago).count()
            
            # Users with emergency stop active
            emergency_stops = db.query(UserPreference).filter(
                UserPreference.emergency_stop == True
            ).count()
            
            # Recently stuck trades (open for >24h)
            day_ago = now - timedelta(days=1)
            stuck_trades = db.query(Trade).filter(
                Trade.opened_at < day_ago,
                Trade.status == 'open'
            ).count()
            
            return {
                "signals_last_hour": signals_last_hour,
                "trades_last_hour": trades_last_hour,
                "total_activity_last_hour": signals_last_hour + trades_last_hour,
                "emergency_stops_active": emergency_stops,
                "stuck_trades_count": stuck_trades,
                "status": "healthy" if stuck_trades < 5 and signals_last_hour > 0 else "degraded"
            }
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {}
