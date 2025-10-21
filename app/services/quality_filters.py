"""
Quality Filter System - Ensures only premium setups get through
Multi-layered filtering for high win-rate signals only
"""
from typing import Dict, Optional, List
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Strict quality filters to ensure only premium setups get broadcast
    - Minimum confidence requirements
    - Session quality gates
    - Multi-confirmation requirements
    - Volume/momentum validation
    """
    
    # Quality thresholds
    MIN_CONFIDENCE = {
        'SCALP': 75,      # Scalp signals need 75%+ confidence
        'SWING': 70,      # Swing signals need 70%+ confidence
        'AGGRESSIVE_SWING': 80  # Aggressive swings need 80%+
    }
    
    MIN_RISK_REWARD = 1.5  # Minimum 1:1.5 R:R ratio
    
    @staticmethod
    def passes_quality_check(signal: Dict) -> tuple[bool, Optional[str]]:
        """
        Comprehensive quality check for signals
        Returns: (passes: bool, rejection_reason: Optional[str])
        """
        
        # 1. SESSION QUALITY FILTER - Block poor sessions entirely
        session = signal.get('session_quality', {})
        if session.get('quality') == 'POOR':
            return False, f"❌ REJECTED: Poor session quality (2-6 AM UTC dead zone)"
        
        # 2. CONFIDENCE THRESHOLD - Category-specific minimums
        category = signal.get('category_name', 'SWING')
        min_conf = QualityFilter.MIN_CONFIDENCE.get(category, 70)
        confidence = signal.get('confidence', 0)
        
        if confidence < min_conf:
            return False, f"❌ REJECTED: Low confidence ({confidence}% < {min_conf}% required for {category})"
        
        # 3. RISK/REWARD RATIO - Minimum 1:1.5
        risk = abs(signal['entry_price'] - signal['stop_loss'])
        reward = abs(signal['take_profit_3'] - signal['entry_price'])
        rr_ratio = reward / risk if risk > 0 else 0
        
        if rr_ratio < QualityFilter.MIN_RISK_REWARD:
            return False, f"❌ REJECTED: Poor R:R ratio ({rr_ratio:.2f} < {QualityFilter.MIN_RISK_REWARD})"
        
        # 4. SIGNAL-SPECIFIC FILTERS
        signal_type = signal.get('signal_type', '')
        
        # FUNDING EXTREME filters
        if signal_type == 'FUNDING_EXTREME':
            funding_rate = abs(signal.get('funding_rate', 0))
            
            # Must be TRULY extreme (>0.1%)
            if funding_rate < 0.1:
                return False, f"❌ REJECTED: Funding rate not extreme enough ({funding_rate:.3f}% < 0.1%)"
            
            # Ultra-extreme funding gets priority (>0.15%)
            if funding_rate > 0.15:
                logger.info(f"✅ PREMIUM: Ultra-extreme funding ({funding_rate:.3f}%)")
        
        # DIVERGENCE filters
        elif 'DIVERGENCE' in signal_type:
            rsi = signal.get('rsi', 50)
            
            # Bullish divergence must be in oversold zone
            if signal['direction'] == 'LONG' and rsi > 40:
                return False, f"❌ REJECTED: Bullish divergence RSI too high ({rsi:.1f} > 40)"
            
            # Bearish divergence must be in overbought zone
            if signal['direction'] == 'SHORT' and rsi < 60:
                return False, f"❌ REJECTED: Bearish divergence RSI too low ({rsi:.1f} < 60)"
        
        # 5. BOOST PREMIUM SETUPS
        # Best session + high confidence = premium setup
        if session.get('quality') == 'BEST' and confidence >= 85:
            logger.info(f"🌟 PREMIUM SETUP: {signal['symbol']} {signal['direction']} ({confidence}% in best session)")
        
        # All checks passed
        return True, None
    
    @staticmethod
    def enhance_signal_quality(signal: Dict) -> Dict:
        """
        Enhance signal with quality scoring and premium flags
        """
        # Calculate quality score (0-100)
        quality_score = 0
        
        # Confidence contribution (40 points max)
        confidence = signal.get('confidence', 0)
        quality_score += min(confidence * 0.4, 40)
        
        # R:R ratio contribution (30 points max)
        risk = abs(signal['entry_price'] - signal['stop_loss'])
        reward = abs(signal['take_profit_3'] - signal['entry_price'])
        rr_ratio = reward / risk if risk > 0 else 0
        quality_score += min(rr_ratio * 10, 30)
        
        # Session quality contribution (20 points max)
        session = signal.get('session_quality', {})
        session_points = {
            'BEST': 20,
            'GOOD': 15,
            'MEDIUM': 10,
            'POOR': 0
        }
        quality_score += session_points.get(session.get('quality', 'MEDIUM'), 10)
        
        # Signal-specific bonus (10 points max)
        if signal.get('signal_type') == 'FUNDING_EXTREME':
            funding_rate = abs(signal.get('funding_rate', 0))
            if funding_rate > 0.15:
                quality_score += 10  # Ultra-extreme funding
            elif funding_rate > 0.12:
                quality_score += 5
        
        elif 'DIVERGENCE' in signal.get('signal_type', ''):
            rsi = signal.get('rsi', 50)
            # Strong divergence in extreme zones
            if (signal['direction'] == 'LONG' and rsi < 30) or \
               (signal['direction'] == 'SHORT' and rsi > 70):
                quality_score += 10
        
        # Add quality metadata
        signal['quality_score'] = min(int(quality_score), 100)
        signal['is_premium'] = quality_score >= 80
        
        # Quality tier
        if quality_score >= 90:
            signal['quality_tier'] = '🌟 PREMIUM'
        elif quality_score >= 75:
            signal['quality_tier'] = '💎 HIGH'
        elif quality_score >= 60:
            signal['quality_tier'] = '✅ GOOD'
        else:
            signal['quality_tier'] = '🟡 STANDARD'
        
        return signal
    
    @staticmethod
    def deduplicate_signals(signals: List[Dict]) -> List[Dict]:
        """
        Remove duplicate/conflicting signals
        - Keep highest quality signal per symbol
        - Remove conflicting directions on same symbol
        """
        if not signals:
            return []
        
        # Group by symbol
        symbol_groups = {}
        for signal in signals:
            symbol = signal['symbol']
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(signal)
        
        # Keep best signal per symbol
        filtered = []
        for symbol, group in symbol_groups.items():
            if len(group) == 1:
                filtered.append(group[0])
            else:
                # Check for conflicting directions
                directions = set(s['direction'] for s in group)
                if len(directions) > 1:
                    logger.warning(f"⚠️ Conflicting signals for {symbol}: {directions}. Keeping highest quality.")
                
                # Sort by quality score and keep best
                best_signal = max(group, key=lambda s: s.get('quality_score', 0))
                filtered.append(best_signal)
                
                logger.info(f"✅ Kept best signal for {symbol}: {best_signal['direction']} (quality: {best_signal.get('quality_score', 0)})")
        
        return filtered


def apply_quality_filters(signals: List[Dict]) -> List[Dict]:
    """
    Main function to apply all quality filters to signals
    Returns only premium setups that pass all checks
    """
    if not signals:
        return []
    
    logger.info(f"📊 Quality filtering {len(signals)} signals...")
    
    # Step 1: Enhance all signals with quality scoring
    enhanced = [QualityFilter.enhance_signal_quality(s) for s in signals]
    
    # Step 2: Apply quality checks
    passed = []
    rejected = []
    
    for signal in enhanced:
        passes, reason = QualityFilter.passes_quality_check(signal)
        
        if passes:
            passed.append(signal)
            logger.info(f"✅ PASSED: {signal['symbol']} {signal['direction']} "
                       f"({signal.get('quality_tier', 'STANDARD')}, "
                       f"score: {signal.get('quality_score', 0)})")
        else:
            rejected.append((signal, reason))
            logger.info(reason)
    
    # Step 3: Deduplicate (keep best per symbol)
    filtered = QualityFilter.deduplicate_signals(passed)
    
    # Summary
    logger.info(f"")
    logger.info(f"📊 QUALITY FILTER SUMMARY:")
    logger.info(f"  Total signals: {len(signals)}")
    logger.info(f"  ✅ Passed: {len(passed)}")
    logger.info(f"  ❌ Rejected: {len(rejected)}")
    logger.info(f"  🎯 Final (deduplicated): {len(filtered)}")
    logger.info(f"  🌟 Premium setups: {sum(1 for s in filtered if s.get('is_premium', False))}")
    logger.info(f"")
    
    return filtered
