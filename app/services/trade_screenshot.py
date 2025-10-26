"""
Trade Screenshot Generator
Creates beautiful shareable images of trade results
"""

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class TradeScreenshotGenerator:
    """Generate beautiful trade summary images for sharing"""
    
    # Color scheme (modern dark theme)
    BG_COLOR = (20, 20, 30)  # Dark navy
    CARD_BG = (30, 35, 50)  # Slightly lighter
    TEXT_PRIMARY = (255, 255, 255)  # White
    TEXT_SECONDARY = (160, 165, 180)  # Light gray
    GREEN = (34, 197, 94)  # Success green
    RED = (239, 68, 68)  # Error red
    ACCENT = (59, 130, 246)  # Blue accent
    GOLD = (251, 191, 36)  # Gold for highlights
    
    def __init__(self):
        self.width = 800
        self.height = 600
        
    def generate_trade_card(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        pnl_percentage: float,
        pnl_amount: float,
        trade_type: str = "DAY_TRADING",
        duration_hours: Optional[float] = None,
        win_streak: int = 0,
        strategy: Optional[str] = None
    ) -> BytesIO:
        """
        Generate a beautiful trade summary image
        
        Returns:
            BytesIO: Image data ready to send via Telegram
        """
        try:
            # Create image
            img = Image.new('RGB', (self.width, self.height), self.BG_COLOR)
            draw = ImageDraw.Draw(img)
            
            # Try to load custom font, fallback to default
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                # Fallback to default font
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Determine if win or loss
            is_win = pnl_percentage > 0
            pnl_color = self.GREEN if is_win else self.RED
            result_emoji = "✅" if is_win else "❌"
            
            # Draw card background
            card_margin = 40
            draw.rounded_rectangle(
                [card_margin, card_margin, self.width - card_margin, self.height - card_margin],
                radius=20,
                fill=self.CARD_BG
            )
            
            # Header: TradehHub AI logo/branding
            y_pos = 60
            draw.text((self.width // 2, y_pos), "TradehHub AI", 
                     font=title_font, fill=self.ACCENT, anchor="mm")
            
            # Symbol and direction
            y_pos = 130
            direction_emoji = "🟢 LONG" if direction.upper() == "LONG" else "🔴 SHORT"
            draw.text((self.width // 2, y_pos), f"{symbol} {direction_emoji}", 
                     font=header_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # PnL - MAIN FOCUS
            y_pos = 200
            pnl_text = f"{result_emoji} {pnl_percentage:+.2f}%"
            draw.text((self.width // 2, y_pos), pnl_text, 
                     font=title_font, fill=pnl_color, anchor="mm")
            
            # USD amount
            y_pos = 250
            usd_text = f"${pnl_amount:+,.2f}"
            draw.text((self.width // 2, y_pos), usd_text, 
                     font=header_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Trade details (2 columns)
            y_pos = 320
            left_x = 150
            right_x = self.width - 150
            
            # Entry price
            draw.text((left_x, y_pos), "Entry", font=small_font, 
                     fill=self.TEXT_SECONDARY, anchor="mm")
            draw.text((left_x, y_pos + 30), f"${entry_price:,.2f}", 
                     font=body_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # Exit price
            draw.text((right_x, y_pos), "Exit", font=small_font, 
                     fill=self.TEXT_SECONDARY, anchor="mm")
            draw.text((right_x, y_pos + 30), f"${exit_price:,.2f}", 
                     font=body_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # Duration and strategy
            y_pos = 400
            if duration_hours is not None:
                if duration_hours < 1:
                    duration_text = f"{int(duration_hours * 60)}m"
                elif duration_hours < 24:
                    duration_text = f"{duration_hours:.1f}h"
                else:
                    duration_text = f"{duration_hours/24:.1f}d"
                
                draw.text((left_x, y_pos), "Duration", font=small_font, 
                         fill=self.TEXT_SECONDARY, anchor="mm")
                draw.text((left_x, y_pos + 30), duration_text, 
                         font=body_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # Strategy
            strategy_display = strategy or ("Top Gainer" if trade_type == "TOP_GAINER" else "Day Trading")
            draw.text((right_x, y_pos), "Strategy", font=small_font, 
                     fill=self.TEXT_SECONDARY, anchor="mm")
            draw.text((right_x, y_pos + 30), strategy_display, 
                     font=body_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # Win streak (if active)
            if win_streak > 0:
                y_pos = 480
                streak_text = f"🔥 {win_streak} Win Streak"
                draw.text((self.width // 2, y_pos), streak_text, 
                         font=body_font, fill=self.GOLD, anchor="mm")
            
            # Footer
            y_pos = self.height - 50
            footer_text = f"Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
            draw.text((self.width // 2, y_pos), footer_text, 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Convert to bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error generating trade screenshot: {e}", exc_info=True)
            raise
    
    def generate_monthly_summary(
        self,
        total_pnl: float,
        total_pnl_pct: float,
        win_rate: float,
        total_trades: int,
        best_trade_pct: float,
        worst_trade_pct: float,
        month_name: str
    ) -> BytesIO:
        """Generate monthly performance summary image"""
        try:
            img = Image.new('RGB', (self.width, self.height), self.BG_COLOR)
            draw = ImageDraw.Draw(img)
            
            # Try to load fonts
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Card background
            card_margin = 40
            draw.rounded_rectangle(
                [card_margin, card_margin, self.width - card_margin, self.height - card_margin],
                radius=20,
                fill=self.CARD_BG
            )
            
            # Header
            y_pos = 60
            draw.text((self.width // 2, y_pos), "TradehHub AI", 
                     font=title_font, fill=self.ACCENT, anchor="mm")
            
            # Month
            y_pos = 120
            draw.text((self.width // 2, y_pos), f"{month_name} Performance", 
                     font=header_font, fill=self.TEXT_PRIMARY, anchor="mm")
            
            # Total PnL
            y_pos = 200
            pnl_color = self.GREEN if total_pnl > 0 else self.RED
            pnl_emoji = "📈" if total_pnl > 0 else "📉"
            pnl_text = f"{pnl_emoji} {total_pnl_pct:+.2f}%"
            draw.text((self.width // 2, y_pos), pnl_text, 
                     font=title_font, fill=pnl_color, anchor="mm")
            
            draw.text((self.width // 2, y_pos + 50), f"${total_pnl:+,.2f}", 
                     font=header_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Stats grid
            y_pos = 320
            left_x = 200
            right_x = self.width - 200
            
            # Win rate
            draw.text((left_x, y_pos), f"{win_rate:.1f}%", 
                     font=header_font, fill=self.GREEN, anchor="mm")
            draw.text((left_x, y_pos + 40), "Win Rate", 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Total trades
            draw.text((right_x, y_pos), str(total_trades), 
                     font=header_font, fill=self.ACCENT, anchor="mm")
            draw.text((right_x, y_pos + 40), "Total Trades", 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Best/Worst
            y_pos = 430
            draw.text((left_x, y_pos), f"{best_trade_pct:+.1f}%", 
                     font=body_font, fill=self.GREEN, anchor="mm")
            draw.text((left_x, y_pos + 30), "Best Trade", 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            draw.text((right_x, y_pos), f"{worst_trade_pct:+.1f}%", 
                     font=body_font, fill=self.RED, anchor="mm")
            draw.text((right_x, y_pos + 30), "Worst Trade", 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Footer
            y_pos = self.height - 50
            footer_text = f"Generated {datetime.utcnow().strftime('%Y-%m-%d')} UTC"
            draw.text((self.width // 2, y_pos), footer_text, 
                     font=small_font, fill=self.TEXT_SECONDARY, anchor="mm")
            
            # Convert to bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error generating monthly summary: {e}", exc_info=True)
            raise


# Global instance
screenshot_generator = TradeScreenshotGenerator()
