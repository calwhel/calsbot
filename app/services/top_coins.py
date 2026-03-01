import logging
import asyncio
from datetime import datetime, timedelta
from typing import Set, Optional
import httpx

logger = logging.getLogger(__name__)

_top_coins_cache: Set[str] = set()
_cache_updated_at: Optional[datetime] = None
_cache_lock = asyncio.Lock()

FALLBACK_TOP_COINS = {
    "BTC", "ETH", "BNB", "XRP", "SOL", "DOGE", "ADA", "TRX", "LINK", "AVAX",
    "SHIB", "TON", "DOT", "BCH", "NEAR", "LTC", "UNI", "ICP", "APT", "PEPE",
    "SUI", "STX", "POL", "FIL", "IMX", "OP", "ARB", "MKR", "ATOM", "HBAR",
    "VET", "AAVE", "RENDER", "FTM", "INJ", "GRT", "ALGO", "SAND", "MANA",
    "AXS", "FLOW", "EOS", "XTZ", "EGLD", "THETA", "CHZ", "WIF", "BONK",
    "FLOKI", "SEI", "TIA", "JUP", "PYTH", "ONDO", "FET", "TAO", "RUNE",
    "PENDLE", "JTO", "W", "BOME", "PEOPLE", "VIRTUAL", "TRUMP", "LAYER",
    "IP", "AIXBT", "TURBO", "ACT", "NEIRO", "GOAT", "PNUT", "MEW", "POPCAT",
    "LDO", "RPL", "CRV", "SNX", "COMP", "SUSHI", "YFI", "1INCH", "BLUR",
    "GMX", "DYDX", "CVX", "BAL", "CAKE", "ENS", "SSV", "LQTY",
    "KAVA", "ZIL", "IOTA", "NEO", "DASH", "ZEC", "XMR", "WAVES", "SC",
    "DCR", "RVN", "BTT", "WIN", "HOT", "CELO", "FTT", "LUNA", "USTC",
    "GALA", "ENJ", "CHR", "ALICE", "DENT", "OGN", "BAND", "OCEAN", "ANKR",
    "STG", "CELR", "COTI", "NMR", "KNC", "REN", "BNT", "UMA", "ZRX",
    "BAT", "STORJ", "IOST", "ONT", "XEM", "LSK", "ARDR", "STEEM", "MINA",
    "ROSE", "ONE", "TFUEL", "SKL", "RAY", "SRM", "STEP", "ORCA", "MNGO",
    "AUDIO", "LRC", "MASK", "IOTX", "SUPER", "POLS", "GHST", "AGIX",
    "CTSI", "BICO", "HIGH", "DUSK", "GLMR", "MOVR", "CFX", "ZEN",
    "REQ", "FARM", "BOND", "FORTH", "INDEX", "DPI", "MVI",
}

COINGECKO_URL = (
    "https://api.coingecko.com/api/v3/coins/markets"
    "?vs_currency=usd&order=market_cap_desc&per_page=160&page=1"
    "&sparkline=false&locale=en"
)


async def _fetch_from_coingecko() -> Set[str]:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(COINGECKO_URL)
            if resp.status_code == 200:
                data = resp.json()
                symbols = {coin["symbol"].upper() for coin in data if coin.get("symbol")}
                logger.info(f"✅ TOP COINS: fetched {len(symbols)} coins from CoinGecko")
                return symbols
    except Exception as e:
        logger.warning(f"TOP COINS: CoinGecko fetch failed: {e}")
    return set()


async def refresh_top_coins() -> None:
    global _top_coins_cache, _cache_updated_at
    async with _cache_lock:
        fetched = await _fetch_from_coingecko()
        if fetched:
            _top_coins_cache = fetched
            _cache_updated_at = datetime.now()
        elif not _top_coins_cache:
            _top_coins_cache = FALLBACK_TOP_COINS.copy()
            _cache_updated_at = datetime.now()
            logger.warning("TOP COINS: Using hardcoded fallback list")


async def get_top_coins() -> Set[str]:
    global _top_coins_cache, _cache_updated_at
    now = datetime.now()
    if (
        not _top_coins_cache
        or _cache_updated_at is None
        or (now - _cache_updated_at) > timedelta(hours=12)
    ):
        await refresh_top_coins()
    return _top_coins_cache


def is_top_coin_sync(symbol: str) -> bool:
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("PERP", "")
    if _top_coins_cache:
        return base in _top_coins_cache
    return base in FALLBACK_TOP_COINS


async def is_top_coin(symbol: str) -> bool:
    coins = await get_top_coins()
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("PERP", "")
    return base in coins
