"""
Trade Screenshot Generator
Creates beautiful shareable images of trade results with custom TradehHub AI background
"""

from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from PIL.Image import Resampling
from io import BytesIO
from datetime import datetime
from typing import Optional
import logging
import os

logger = logging.getLogger(__name__)


class TradeScreenshotGenerator:
    """Generate beautiful trade summary images for sharing"""
    
    # Color scheme (overlays on custom background)
    TEXT_PRIMARY = (255, 255, 255)  # White
    TEXT_SECONDARY = (200, 210, 220)  # Light gray
    GREEN = (34, 197, 94)  # Success green
    RED = (239, 68, 68)  # Error red
    ACCENT = (59, 130, 246)  # Blue accent
    GOLD = (251, 191, 36)  # Gold for highlights
    CYAN = (34, 211, 238)  # Cyan (matches robot eyes)
    
    def __init__(self):
        self.width = 1024
        self.height = 768
        self.background_path = "app/assets/trade_card_background.png"
        
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
        Generate a beautiful trade summary image with custom background
        
        Returns:
            BytesIO: Image data ready to send via Telegram
        """
        try:
            # Load custom background
            if os.path.exists(self.background_path):
                img = Image.open(self.background_path).convert('RGB')
                # Resize to standard dimensions if needed
                if img.size != (self.width, self.height):
                    img = img.resize((self.width, self.height), Resampling.LANCZOS)
            else:
                # Fallback to dark gradient if background not found
                logger.warning(f"Background not found at {self.background_path}, using fallback")
                img = Image.new('RGB', (self.width, self.height), (20, 30, 40))
            
            draw = ImageDraw.Draw(img)
            
            # Load fonts
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Determine if win or loss
            is_win = pnl_percentage > 0
            pnl_color = self.GREEN if is_win else self.RED
            result_emoji = "✅" if is_win else "❌"
            
            # Add semi-transparent overlay on LEFT side for better text readability
            # (Robot is on the right, so we put text on the left)
            overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Dark overlay on left half for text
            overlay_draw.rectangle(
                [0, 0, self.width // 2 + 100, self.height],
                fill=(10, 20, 30, 180)  # Dark with 70% opacity
            )
            
            # Blend overlay
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Text positioning (LEFT SIDE)
            left_margin = 60
            center_x = self.width // 4  # Center of left half
            
            # Symbol and direction at top
            y_pos = 80
            direction_emoji = "🟢" if direction.upper() == "LONG" else "🔴"
            symbol_text = f"{symbol} {direction_emoji} {direction.upper()}"
            draw.text((left_margin, y_pos), symbol_text, 
                     font=header_font, fill=self.CYAN)
            
            # PnL - MAIN FOCUS (large and centered on left)
            y_pos = 200
            pnl_text = f"{pnl_percentage:+.2f}%"
            draw.text((left_margin, y_pos), pnl_text, 
                     font=title_font, fill=pnl_color)
            
            # Result emoji next to PnL
            draw.text((left_margin + 280, y_pos + 10), result_emoji, 
                     font=header_font, fill=pnl_color)
            
            # USD amount
            y_pos = 290
            usd_text = f"${pnl_amount:+,.2f} USD"
            draw.text((left_margin, y_pos), usd_text, 
                     font=body_font, fill=self.TEXT_SECONDARY)
            
            # Trade details section
            y_pos = 380
            
            # Entry price
            draw.text((left_margin, y_pos), "ENTRY", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            draw.text((left_margin, y_pos + 35), f"${entry_price:,.4f}", 
                     font=body_font, fill=self.TEXT_PRIMARY)
            
            # Exit price
            y_pos = 470
            draw.text((left_margin, y_pos), "EXIT", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            draw.text((left_margin, y_pos + 35), f"${exit_price:,.4f}", 
                     font=body_font, fill=self.TEXT_PRIMARY)
            
            # Duration
            y_pos = 560
            if duration_hours is not None:
                if duration_hours < 1:
                    duration_text = f"{int(duration_hours * 60)}m"
                elif duration_hours < 24:
                    duration_text = f"{duration_hours:.1f}h"
                else:
                    duration_text = f"{duration_hours/24:.1f}d"
                
                draw.text((left_margin, y_pos), "DURATION", 
                         font=small_font, fill=self.TEXT_SECONDARY)
                draw.text((left_margin, y_pos + 35), duration_text, 
                         font=body_font, fill=self.TEXT_PRIMARY)
            
            # Win streak badge (if active)
            if win_streak > 0:
                y_pos = 650
                streak_text = f"🔥 {win_streak} WIN STREAK"
                draw.text((left_margin, y_pos), streak_text, 
                         font=body_font, fill=self.GOLD)
            
            # Strategy tag (top right corner)
            strategy_display = strategy or ("Top Gainer" if trade_type == "TOP_GAINER" else "Day Trading")
            draw.text((self.width - 60, 30), strategy_display.upper(), 
                     font=small_font, fill=self.CYAN, anchor="rt")
            
            # Timestamp (bottom left)
            timestamp_text = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
            draw.text((left_margin, self.height - 40), timestamp_text, 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            # Convert to bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG', quality=95)
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
        """Generate monthly performance summary image with custom background"""
        try:
            # Load custom background
            if os.path.exists(self.background_path):
                img = Image.open(self.background_path).convert('RGB')
                if img.size != (self.width, self.height):
                    img = img.resize((self.width, self.height), Resampling.LANCZOS)
            else:
                img = Image.new('RGB', (self.width, self.height), (20, 30, 40))
            
            draw = ImageDraw.Draw(img)
            
            # Load fonts
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
                header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                title_font = ImageFont.load_default()
                header_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Add overlay
            overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [0, 0, self.width // 2 + 100, self.height],
                fill=(10, 20, 30, 180)
            )
            
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Layout
            left_margin = 60
            
            # Month header
            y_pos = 80
            draw.text((left_margin, y_pos), f"{month_name.upper()}", 
                     font=header_font, fill=self.CYAN)
            draw.text((left_margin, y_pos + 60), "PERFORMANCE", 
                     font=body_font, fill=self.TEXT_SECONDARY)
            
            # Total PnL
            y_pos = 200
            pnl_color = self.GREEN if total_pnl > 0 else self.RED
            pnl_text = f"{total_pnl_pct:+.2f}%"
            draw.text((left_margin, y_pos), pnl_text, 
                     font=title_font, fill=pnl_color)
            
            y_pos = 280
            draw.text((left_margin, y_pos), f"${total_pnl:+,.2f} USD", 
                     font=body_font, fill=self.TEXT_SECONDARY)
            
            # Stats
            y_pos = 370
            
            # Win rate
            draw.text((left_margin, y_pos), "WIN RATE", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            draw.text((left_margin, y_pos + 35), f"{win_rate:.1f}%", 
                     font=header_font, fill=self.GREEN)
            
            # Total trades
            y_pos = 480
            draw.text((left_margin, y_pos), "TOTAL TRADES", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            draw.text((left_margin, y_pos + 35), str(total_trades), 
                     font=header_font, fill=self.CYAN)
            
            # Best/Worst
            y_pos = 590
            draw.text((left_margin, y_pos), 
                     f"Best: {best_trade_pct:+.1f}%  |  Worst: {worst_trade_pct:+.1f}%", 
                     font=body_font, fill=self.TEXT_SECONDARY)
            
            # Timestamp
            timestamp_text = datetime.utcnow().strftime('%Y-%m-%d UTC')
            draw.text((left_margin, self.height - 40), timestamp_text, 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            # Convert to bytes
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG', quality=95)
            img_bytes.seek(0)
            
            return img_bytes
            
        except Exception as e:
            logger.error(f"Error generating monthly summary: {e}", exc_info=True)
            raise


# Global instance
screenshot_generator = TradeScreenshotGenerator()
