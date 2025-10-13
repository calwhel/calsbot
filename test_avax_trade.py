import asyncio
import ccxt.async_support as ccxt
from app.database import SessionLocal
from app.models import User, UserPreference, Signal, Trade
from app.utils.encryption import decrypt_api_key
from datetime import datetime

async def test_avax_trade():
    db = SessionLocal()
    try:
        # Get user and preferences
        user = db.query(User).filter_by(telegram_id='5603353066').first()
        prefs = user.preferences
        
        # Get AVAX signal
        signal = db.query(Signal).filter_by(id=21).first()
        
        print("=" * 50)
        print("🧪 AVAX TEST TRADE")
        print("=" * 50)
        print(f"User: {user.username}")
        print(f"Balance to use: Your MEXC Futures wallet")
        print(f"Position Size: {prefs.position_size_percent}%")
        print(f"Leverage: {prefs.user_leverage}x")
        print()
        print(f"Signal: {signal.symbol} {signal.direction}")
        print(f"Entry: ${signal.entry_price}")
        print(f"Stop Loss: ${signal.stop_loss} (-4.4%)")
        print(f"TP1: ${signal.take_profit_1} (+5.3%)")
        print(f"TP2: ${signal.take_profit_2} (+10.5%)")
        print(f"TP3: ${signal.take_profit_3} (+17.5%)")
        print(f"Risk:Reward = 1:4.0")
        print()
        
        # Decrypt API keys
        api_key = decrypt_api_key(prefs.mexc_api_key)
        api_secret = decrypt_api_key(prefs.mexc_api_secret)
        
        # Initialize MEXC
        exchange = ccxt.mexc({
            'apiKey': api_key,
            'secret': api_secret,
            'options': {'defaultType': 'swap'}
        })
        
        # Get balance
        print("📊 Checking account balance...")
        balance_info = await exchange.fetch_balance()
        usdt_balance = balance_info.get('USDT', {}).get('free', 0.0)
        print(f"✅ USDT Balance: ${usdt_balance:.2f}")
        
        if usdt_balance < 1:
            print("❌ Insufficient balance for test trade")
            await exchange.close()
            return
        
        # Calculate position using user's settings
        position_percent = prefs.position_size_percent
        position_size_usd = (usdt_balance * position_percent) / 100
        amount = position_size_usd / signal.entry_price
        
        print(f"\n💰 Position Sizing:")
        print(f"Position Size: ${position_size_usd:.2f} ({position_percent}%)")
        print(f"Amount: {amount:.4f} AVAX")
        print(f"With 10x leverage: ${position_size_usd * 10:.2f} buying power")
        
        print(f"\n🚀 Placing TEST order on MEXC...")
        
        # Set leverage
        await exchange.set_leverage(
            10,
            'AVAX/USDT:USDT',
            params={
                'openType': 2,  # Cross margin
                'positionType': 1  # Long
            }
        )
        print("✅ Leverage set to 10x")
        
        # Place market order
        order = await exchange.create_market_order(
            symbol='AVAX/USDT:USDT',
            side='buy',
            amount=amount,
            params={'positionSide': 'long'}
        )
        
        print(f"✅ ORDER PLACED!")
        print(f"Order ID: {order.get('id')}")
        print(f"Status: {order.get('status')}")
        
        # Save to database
        trade = Trade(
            user_id=user.id,
            signal_id=signal.id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            position_size=position_size_usd,
            remaining_size=amount,
            status='open',
            opened_at=datetime.utcnow()
        )
        db.add(trade)
        db.commit()
        
        print(f"\n✅ Trade saved to database (ID: {trade.id})")
        print(f"\n🎯 Now monitoring position for:")
        print(f"  - TP1 @ ${signal.take_profit_1} (30% close)")
        print(f"  - TP2 @ ${signal.take_profit_2} (60% close)")
        print(f"  - TP3 @ ${signal.take_profit_3} (100% close)")
        print(f"  - Trailing stop will activate at 2% profit")
        
        await exchange.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    asyncio.run(test_avax_trade())
