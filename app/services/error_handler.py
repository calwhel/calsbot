import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, DateTime, Text
from app.database import Base, SessionLocal

logger = logging.getLogger(__name__)


class ErrorLog(Base):
    """Track errors for admin monitoring and debugging"""
    __tablename__ = "error_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True, index=True)
    error_type = Column(String, index=True)  # 'trade_execution', 'signal_generation', 'api_error', etc.
    error_message = Column(Text)
    stack_trace = Column(Text, nullable=True)
    severity = Column(String, default='warning')  # 'info', 'warning', 'error', 'critical'
    context = Column(Text, nullable=True)  # JSON string with additional context
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    resolved = Column(Integer, default=0)  # 0 = not resolved, 1 = resolved


class ErrorHandler:
    """Centralized error handling and logging system"""
    
    # In-memory error rate tracking (prevents DB spam)
    _error_counts = defaultdict(int)
    _last_reset = datetime.utcnow()
    _error_throttle = defaultdict(datetime.utcnow)  # Last logged time per error type
    _throttle_seconds = 60  # Only log each error type once per minute during bursts
    
    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        user_id: Optional[int] = None,
        severity: str = 'warning',
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None
    ):
        """
        Log an error to database and system logs with rate limiting
        
        Args:
            error_type: Category of error (e.g., 'trade_execution', 'api_error')
            error_message: Human-readable error description
            user_id: User ID if error is user-specific
            severity: 'info', 'warning', 'error', 'critical'
            exception: The caught exception (for stack trace)
            context: Additional context as dict
        """
        try:
            now = datetime.utcnow()
            
            # Update in-memory error rate
            ErrorHandler._error_counts[error_type] += 1
            
            # THROTTLE DATABASE WRITES: Skip DB write if same error type logged recently
            # This prevents DB spam during API outages/bursts
            throttle_key = f"{error_type}_{user_id or 'system'}"
            last_logged = ErrorHandler._error_throttle.get(throttle_key)
            
            should_log_to_db = True
            if last_logged:
                time_since_last = (now - last_logged).total_seconds()
                if time_since_last < ErrorHandler._throttle_seconds:
                    should_log_to_db = False  # Skip DB write, too soon
            
            # Always log to system logger (cheap)
            log_message = f"[{error_type}] {error_message}"
            if user_id:
                log_message += f" (User: {user_id})"
            
            if severity == 'critical':
                logger.critical(log_message)
            elif severity == 'error':
                logger.error(log_message)
            elif severity == 'warning':
                logger.warning(log_message)
            else:
                logger.info(log_message)
            
            # Save to database only if not throttled
            if should_log_to_db:
                # Get stack trace if exception provided
                stack_trace = None
                if exception:
                    stack_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
                
                db = SessionLocal()
                try:
                    error_log = ErrorLog(
                        user_id=user_id,
                        error_type=error_type,
                        error_message=error_message[:500],  # Truncate long messages
                        stack_trace=stack_trace,
                        severity=severity,
                        context=str(context) if context else None
                    )
                    db.add(error_log)
                    db.commit()
                    
                    # Update throttle timestamp
                    ErrorHandler._error_throttle[throttle_key] = now
                finally:
                    db.close()
                
        except Exception as e:
            # Fallback - don't let error handler crash
            logger.error(f"Error in error handler: {e}")
    
    @staticmethod
    def get_error_rate(error_type: Optional[str] = None, hours: int = 1) -> int:
        """Get error count for a specific type or all errors"""
        # Reset counter if needed
        now = datetime.utcnow()
        if (now - ErrorHandler._last_reset).total_seconds() > 3600:
            ErrorHandler._error_counts.clear()
            ErrorHandler._last_reset = now
        
        if error_type:
            return ErrorHandler._error_counts.get(error_type, 0)
        else:
            return sum(ErrorHandler._error_counts.values())
    
    @staticmethod
    def get_recent_errors(db: Session, hours: int = 24, limit: int = 50) -> List[ErrorLog]:
        """Get recent errors from database"""
        try:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            errors = db.query(ErrorLog).filter(
                ErrorLog.created_at >= cutoff
            ).order_by(ErrorLog.created_at.desc()).limit(limit).all()
            return errors
        except Exception as e:
            logger.error(f"Error fetching recent errors: {e}")
            return []
    
    @staticmethod
    def get_error_stats(db: Session, hours: int = 24) -> Dict:
        """Get error statistics for admin dashboard"""
        try:
            from sqlalchemy import func
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            # Total errors
            total_errors = db.query(ErrorLog).filter(ErrorLog.created_at >= cutoff).count()
            
            # Errors by severity
            critical_errors = db.query(ErrorLog).filter(
                ErrorLog.created_at >= cutoff,
                ErrorLog.severity == 'critical'
            ).count()
            
            errors_by_severity = db.query(
                ErrorLog.severity,
                func.count(ErrorLog.id).label('count')
            ).filter(
                ErrorLog.created_at >= cutoff
            ).group_by(ErrorLog.severity).all()
            
            # Errors by type
            errors_by_type = db.query(
                ErrorLog.error_type,
                func.count(ErrorLog.id).label('count')
            ).filter(
                ErrorLog.created_at >= cutoff
            ).group_by(ErrorLog.error_type).order_by(func.count(ErrorLog.id).desc()).limit(5).all()
            
            # Error rate (per hour)
            error_rate = total_errors / hours if hours > 0 else 0
            
            return {
                "total_errors": total_errors,
                "critical_errors": critical_errors,
                "error_rate_per_hour": error_rate,
                "by_severity": {sev: count for sev, count in errors_by_severity},
                "top_error_types": {err_type: count for err_type, count in errors_by_type}
            }
        except Exception as e:
            logger.error(f"Error getting error stats: {e}")
            return {}
    
    @staticmethod
    def get_user_friendly_message(error_type: str, original_error: str) -> str:
        """Convert technical errors to user-friendly messages"""
        
        error_messages = {
            "insufficient_balance": "💰 Insufficient balance to execute this trade. Please add more funds or reduce your position size.",
            "api_key_invalid": "🔑 Your API key is invalid or has been revoked. Please update your exchange API credentials in /settings.",
            "api_key_expired": "⏰ Your API key has expired. Please generate a new key from your exchange and update it.",
            "rate_limit": "⏱️ Too many requests to the exchange. Please wait a moment and try again.",
            "network_error": "🌐 Network connection issue. Please check your internet connection and try again.",
            "exchange_maintenance": "🔧 The exchange is currently under maintenance. Please try again later.",
            "symbol_not_found": "❌ This trading pair is not available on your exchange. Please check the symbol.",
            "multi_analysis_failed": "📊 Signal failed multi-analysis validation. This trade was automatically skipped to protect your capital.",
            "max_positions": "📈 You've reached your maximum number of open positions. Close some positions or increase your limit in /settings.",
            "correlation_blocked": "⚠️ Trade blocked due to correlation filter. You already have a similar position open.",
            "daily_loss_limit": "🛑 Daily loss limit reached. Auto-trading paused to protect your capital. Limit will reset tomorrow.",
            "emergency_stop": "🚨 Emergency stop is active. All trading is paused. Use /resume_trading to continue.",
        }
        
        # Return friendly message if we have one
        for key, message in error_messages.items():
            if key in error_type.lower() or key in original_error.lower():
                return message
        
        # Fallback to generic message
        return f"⚠️ An error occurred: {original_error[:100]}. Please try again or contact support if this persists."


# Retry decorator for critical operations
def with_retry(max_attempts: int = 3, delay_seconds: int = 2, exponential_backoff: bool = True):
    """
    Decorator to retry failed operations with exponential backoff
    
    Usage:
        @with_retry(max_attempts=3)
        async def execute_trade(...):
            ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            import asyncio
            
            last_exception = None
            delay = delay_seconds
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_attempts - 1:
                        logger.warning(f"{func.__name__} failed (attempt {attempt + 1}/{max_attempts}): {e}")
                        await asyncio.sleep(delay)
                        
                        if exponential_backoff:
                            delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator
