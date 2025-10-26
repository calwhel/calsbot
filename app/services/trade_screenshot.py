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
        """Generate clean TradehHub-style trade card (like Bitunix reference)"""
        try:
            # Load custom background
            if os.path.exists(self.background_path):
                img = Image.open(self.background_path).convert('RGB')
                if img.size != (self.width, self.height):
                    img = img.resize((self.width, self.height), Resampling.LANCZOS)
            else:
                logger.warning(f"Background not found at {self.background_path}, using fallback")
                img = Image.new('RGB', (self.width, self.height), (20, 30, 40))
            
            draw = ImageDraw.Draw(img)
            
            # Fonts - MASSIVE PnL focus
            try:
                massive_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 140)  # HUGE PnL
                large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 56)    # Symbol
                medium_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)   # Prices
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)         # Labels
                tiny_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)          # Bottom
            except:
                massive_font = large_font = medium_font = small_font = tiny_font = ImageFont.load_default()
            
            # Dark overlay on left for text
            overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [0, 0, self.width // 2 + 100, self.height],
                fill=(5, 10, 15, 210)
            )
            
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            left_margin = 50
            pnl_color = self.GREEN if pnl_percentage > 0 else self.RED
            direction_color = self.GREEN if direction.upper() == "LONG" else self.RED
            
            # Symbol and direction at top (like "AVAXUSDT" in reference)
            draw.text((left_margin, 50), symbol, 
                     font=large_font, fill=self.TEXT_PRIMARY)
            
            # Long/Short with leverage (like "Long | 15X" in reference)
            y_pos = 115
            direction_text = f"{direction.capitalize()} | 10X"
            draw.text((left_margin, y_pos), direction_text, 
                     font=small_font, fill=direction_color)
            
            # MASSIVE PnL percentage (HERO ELEMENT)
            y_pos = 200
            pnl_text = f"{pnl_percentage:+.2f}%"
            draw.text((left_margin, y_pos), pnl_text, 
                     font=massive_font, fill=pnl_color)
            
            # Dollar amount
            y_pos = 360
            draw.text((left_margin, y_pos), f"${pnl_amount:+,.2f} USD", 
                     font=medium_font, fill=self.TEXT_PRIMARY)
            
            # Entry Price (like reference)
            y_pos = 440
            draw.text((left_margin, y_pos), f"Entry Price  {entry_price:,.4f}", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            # Exit Price (like "Last Price" in reference)
            y_pos = 480
            draw.text((left_margin, y_pos), f"Last Price  {exit_price:,.4f}", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            # Duration (if available)
            if duration_hours is not None:
                y_pos = 520
                if duration_hours < 1:
                    duration_text = f"{int(duration_hours * 60)}m"
                elif duration_hours < 24:
                    duration_text = f"{duration_hours:.1f}h"
                else:
                    duration_text = f"{duration_hours/24:.1f}d"
                draw.text((left_margin, y_pos), f"Duration  {duration_text}", 
                         font=small_font, fill=self.TEXT_SECONDARY)
            
            # Win streak (if active)
            if win_streak > 0:
                y_pos = 560 if duration_hours else 520
                draw.text((left_margin, y_pos), f"🔥 {win_streak} Win Streak", 
                         font=small_font, fill=self.GOLD)
            
            # Bottom branding
            y_pos = self.height - 80
            draw.text((left_margin, y_pos), "Fully Automated Trading", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            # Referral code
            y_pos = self.height - 40
            draw.text((left_margin, y_pos), "Referral code: tradehub", 
                     font=tiny_font, fill=self.TEXT_SECONDARY)
            
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
        month_name: str,
        referral_code: str = "tradehub"
    ) -> BytesIO:
        """Generate PnL card matching EXACT Bitunix style with TradehHub branding"""
        try:
            # Load custom background
            if os.path.exists(self.background_path):
                img = Image.open(self.background_path).convert('RGB')
                if img.size != (self.width, self.height):
                    img = img.resize((self.width, self.height), Resampling.LANCZOS)
            else:
                img = Image.new('RGB', (self.width, self.height), (20, 30, 40))
            
            draw = ImageDraw.Draw(img)
            
            # Fonts - EXACTLY like Bitunix with MASSIVE ROI
            try:
                mega_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 200)   # MASSIVE ROI
                large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)    # Period name
                medium_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)        # Stats
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)         # Bottom text
            except:
                mega_font = large_font = medium_font = small_font = ImageFont.load_default()
            
            # Darker overlay on left (like Bitunix)
            overlay = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [0, 0, self.width // 2 + 100, self.height],
                fill=(5, 10, 15, 220)
            )
            
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
            draw = ImageDraw.Draw(img)
            
            left_margin = 50
            pnl_color = self.GREEN if total_pnl_pct > 0 else self.RED
            
            # Period name at top (small, minimal)
            draw.text((left_margin, 65), month_name.upper(), 
                     font=large_font, fill=self.TEXT_PRIMARY)
            
            # ⭐ MASSIVE ROI PERCENTAGE - THE STAR OF THE SHOW ⭐
            # This is the main focal point - huge, bold, impossible to miss
            y_pos = 180
            pnl_text = f"{total_pnl_pct:+.2f}%"
            draw.text((left_margin, y_pos), pnl_text, 
                     font=mega_font, fill=pnl_color)
            
            # Secondary stats below (smaller, supporting info)
            y_pos = 425
            draw.text((left_margin, y_pos), f"Total PnL  ${total_pnl:+,.2f}", 
                     font=medium_font, fill=self.TEXT_SECONDARY)
            
            y_pos += 60
            draw.text((left_margin, y_pos), f"Win Rate  {win_rate:.1f}%", 
                     font=medium_font, fill=self.TEXT_SECONDARY)
            
            y_pos += 60
            draw.text((left_margin, y_pos), f"Total Trades  {total_trades}", 
                     font=medium_font, fill=self.TEXT_SECONDARY)
            
            # Bottom branding - split into two lines for better layout
            y_pos = self.height - 110
            draw.text((left_margin, y_pos), "FULLY AUTOMATED TRADING", 
                     font=small_font, fill=self.TEXT_SECONDARY)
            
            y_pos = self.height - 75
            draw.text((left_margin, y_pos), f"POWERED BY TRADEHUB AI", 
                     font=small_font, fill=self.CYAN)
            
            # Referral code
            y_pos = self.height - 40
            draw.text((left_margin, y_pos), f"Referral code: {referral_code}", 
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
