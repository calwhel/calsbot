from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc, asc, func, case
from datetime import datetime, timedelta
from typing import Optional
import httpx
import asyncio
import logging

from app.database import get_db
from app.models import Trade, Signal

router = APIRouter()
logger = logging.getLogger(__name__)

_price_cache = {}
_price_cache_ttl = 15

async def _fetch_live_prices(symbols: list[str]) -> dict:
    """Fetch live prices from multiple sources (Binance Futures + MEXC) for verification."""
    global _price_cache
    now = datetime.utcnow().timestamp()
    
    needs_fetch = False
    for s in symbols:
        clean = s.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
        if not clean.endswith('USDT'):
            clean += 'USDT'
        cached = _price_cache.get(clean)
        if not cached or (now - cached[1]) > _price_cache_ttl:
            needs_fetch = True
            break
    
    if needs_fetch:
        async with httpx.AsyncClient(timeout=5) as client:
            tasks = []
            async def fetch_binance():
                try:
                    resp = await client.get("https://fapi.binance.com/fapi/v1/ticker/price")
                    if resp.status_code == 200:
                        for item in resp.json():
                            sym = item.get("symbol", "")
                            price = float(item.get("price", 0))
                            if price > 0:
                                _price_cache[sym] = (price, now)
                except Exception as e:
                    logger.warning(f"Binance price fetch failed: {e}")
            
            async def fetch_mexc():
                try:
                    resp = await client.get("https://api.mexc.com/api/v3/ticker/price")
                    if resp.status_code == 200:
                        for item in resp.json():
                            sym = item.get("symbol", "")
                            price = float(item.get("price", 0))
                            if price > 0 and sym not in _price_cache:
                                _price_cache[sym] = (price, now)
                except Exception as e:
                    logger.warning(f"MEXC price fetch failed: {e}")
            
            await asyncio.gather(fetch_binance(), fetch_mexc())
    
    result = {}
    for s in symbols:
        clean = s.replace('/', '').replace(':USDT', '').replace('-USDT', '').upper()
        if not clean.endswith('USDT'):
            clean += 'USDT'
        cached = _price_cache.get(clean)
        if cached:
            result[s] = cached[0]
    return result

TRACKER_START_DATE = datetime(2026, 2, 3)

ALLOWED_SORT_COLS = {"opened_at", "symbol", "direction", "entry_price", "exit_price", "pnl", "pnl_percent"}


@router.get("/api/trades")
async def get_trades(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=10, le=200),
    status: Optional[str] = Query(None),
    direction: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
    trade_type: Optional[str] = Query(None),
    sort_by: str = Query("opened_at"),
    sort_dir: str = Query("desc"),
    days: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(Trade).filter(Trade.status != 'failed').filter(Trade.opened_at >= TRACKER_START_DATE)

    if status and status != "all":
        if status == "closed":
            query = query.filter(Trade.status.in_(['closed', 'tp_hit', 'sl_hit']))
        else:
            query = query.filter(Trade.status == status)
    if direction and direction != "all":
        query = query.filter(Trade.direction == direction)
    if symbol:
        query = query.filter(Trade.symbol.ilike(f"%{symbol}%"))
    if trade_type and trade_type != "all":
        query = query.filter(Trade.trade_type == trade_type)
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.filter(Trade.opened_at >= cutoff)

    total = query.count()

    safe_sort = sort_by if sort_by in ALLOWED_SORT_COLS else "opened_at"
    sort_col = getattr(Trade, safe_sort, Trade.opened_at)
    if sort_dir == "asc":
        query = query.order_by(asc(sort_col))
    else:
        query = query.order_by(desc(sort_col))

    trades = query.offset((page - 1) * per_page).limit(per_page).all()

    open_trades = [t for t in trades if t.status == 'open']
    live_prices = {}
    if open_trades:
        symbols = list(set(t.symbol for t in open_trades))
        live_prices = await _fetch_live_prices(symbols)

    results = []
    for t in trades:
        ticker = t.symbol.replace("USDT", "").replace("/USDT:USDT", "").replace("/", "")
        
        pnl = t.pnl or 0
        pnl_pct = t.pnl_percent or 0
        current_price = None
        
        if t.status == 'open' and t.entry_price and t.entry_price > 0:
            live_price = live_prices.get(t.symbol)
            if live_price:
                current_price = live_price
                leverage = t.leverage or 1
                if t.direction == 'LONG':
                    pnl_pct = ((live_price - t.entry_price) / t.entry_price) * 100 * leverage
                else:
                    pnl_pct = ((t.entry_price - live_price) / t.entry_price) * 100 * leverage
                size = t.position_size or 0
                pnl = size * (pnl_pct / 100)
                if pnl_pct >= 100 and not t.breakeven_moved:
                    try:
                        t.breakeven_moved = True
                        t.stop_loss = t.entry_price
                        db.commit()
                        logger.info(f"AUTO-BREAKEVEN: {t.symbol} hit {pnl_pct:.1f}% ROI - SL moved to entry {t.entry_price}")
                    except Exception as e:
                        logger.error(f"Failed to set breakeven for {t.symbol}: {e}")
                        db.rollback()
        
        effective_pnl = pnl
        if t.status != 'open':
            effective_pnl = t.pnl or 0

        if t.status == 'tp_hit':
            result_label = "TP HIT"
        elif t.status == 'sl_hit':
            if t.breakeven_moved and abs(t.pnl_percent or 0) < 10:
                result_label = "BREAKEVEN"
            else:
                result_label = "SL HIT"
        elif t.status == 'open':
            if pnl_pct > 0:
                result_label = "RUNNING +"
            elif pnl_pct < 0:
                result_label = "RUNNING -"
            else:
                result_label = "OPEN"
        elif t.breakeven_moved and abs(effective_pnl) < 1 and abs(t.pnl_percent or 0) < 10:
            result_label = "BREAKEVEN"
        elif effective_pnl > 0:
            result_label = "WIN"
        elif effective_pnl < 0:
            result_label = "LOSS"
        else:
            result_label = "BREAKEVEN"

        results.append({
            "id": t.id,
            "symbol": f"${ticker}",
            "raw_symbol": t.symbol,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "current_price": current_price,
            "stop_loss": t.stop_loss,
            "tp1": t.take_profit_1,
            "tp2": t.take_profit_2,
            "tp3": t.take_profit_3,
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl_pct, 2),
            "position_size": round(t.position_size or 0, 2),
            "status": t.status,
            "result": result_label,
            "tp1_hit": t.tp1_hit or False,
            "tp2_hit": t.tp2_hit or False,
            "tp3_hit": t.tp3_hit or False,
            "leverage": t.leverage,
            "breakeven_moved": t.breakeven_moved or False,
            "trade_type": t.trade_type or "STANDARD",
            "opened_at": t.opened_at.isoformat() if t.opened_at else None,
            "closed_at": t.closed_at.isoformat() if t.closed_at else None,
            "duration_mins": round((t.closed_at - t.opened_at).total_seconds() / 60, 1) if t.closed_at and t.opened_at else None,
        })

    return JSONResponse({
        "trades": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    })


@router.get("/api/trades/stats")
async def get_trade_stats(
    days: Optional[int] = Query(None),
    db: Session = Depends(get_db)
):
    from sqlalchemy import text
    
    params = {"start_date": TRACKER_START_DATE}
    date_filter = ""
    if days:
        date_filter = " AND opened_at >= :cutoff"
        params["cutoff"] = datetime.utcnow() - timedelta(days=int(days))

    sql = text(f"""
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE COALESCE(pnl, 0) > 0) as wins,
            COUNT(*) FILTER (WHERE COALESCE(pnl, 0) < 0 AND NOT (COALESCE(breakeven_moved, false) AND ABS(COALESCE(pnl_percent, 0)) < 10)) as losses,
            COUNT(*) FILTER (WHERE COALESCE(pnl, 0) = 0 OR (COALESCE(breakeven_moved, false) AND ABS(COALESCE(pnl_percent, 0)) < 10)) as breakeven,
            ROUND(COALESCE(AVG(pnl_percent), 0)::numeric, 2) as avg_roi,
            ROUND(COALESCE(AVG(pnl_percent) FILTER (WHERE COALESCE(pnl, 0) > 0), 0)::numeric, 2) as avg_win_roi,
            ROUND(COALESCE(AVG(pnl_percent) FILTER (WHERE COALESCE(pnl, 0) < 0), 0)::numeric, 2) as avg_loss_roi,
            ROUND(COALESCE(MAX(pnl_percent), 0)::numeric, 2) as best_roi,
            ROUND(COALESCE(SUM(pnl_percent), 0)::numeric, 2) as total_roi
        FROM trades WHERE status IN ('closed', 'tp_hit', 'sl_hit') AND opened_at >= :start_date{date_filter}
    """)
    row = db.execute(sql, params).fetchone()

    if not row or row[0] == 0:
        return JSONResponse({"total": 0, "wins": 0, "losses": 0, "breakeven": 0,
                             "win_rate": 0, "avg_roi": 0,
                             "avg_win_roi": 0, "avg_loss_roi": 0, "best_roi": 0,
                             "total_roi": 0, "by_type": {}, "by_direction": {}})

    total = int(row[0])
    wins = int(row[1])

    type_sql = text(f"""
        SELECT COALESCE(trade_type, 'STANDARD') as tt,
               COUNT(*) as cnt,
               COUNT(*) FILTER (WHERE COALESCE(pnl, 0) > 0) as w,
               ROUND(COALESCE(AVG(pnl_percent), 0)::numeric, 2) as avg_roi
        FROM trades WHERE status IN ('closed', 'tp_hit', 'sl_hit') AND opened_at >= :start_date{date_filter}
        GROUP BY COALESCE(trade_type, 'STANDARD')
    """)
    by_type = {}
    for r in db.execute(type_sql, params).fetchall():
        c = int(r[1])
        by_type[r[0]] = {
            "count": c, "wins": int(r[2]), "avg_roi": float(r[3]),
            "win_rate": round(int(r[2]) / c * 100, 1) if c else 0
        }

    dir_sql = text(f"""
        SELECT COALESCE(direction, 'UNKNOWN') as d,
               COUNT(*) as cnt,
               COUNT(*) FILTER (WHERE COALESCE(pnl, 0) > 0) as w,
               ROUND(COALESCE(AVG(pnl_percent), 0)::numeric, 2) as avg_roi
        FROM trades WHERE status IN ('closed', 'tp_hit', 'sl_hit') AND opened_at >= :start_date{date_filter}
        GROUP BY COALESCE(direction, 'UNKNOWN')
    """)
    by_dir = {}
    for r in db.execute(dir_sql, params).fetchall():
        c = int(r[1])
        by_dir[r[0]] = {
            "count": c, "wins": int(r[2]), "avg_roi": float(r[3]),
            "win_rate": round(int(r[2]) / c * 100, 1) if c else 0
        }

    return JSONResponse({
        "total": total,
        "wins": wins,
        "losses": int(row[2]),
        "breakeven": int(row[3]),
        "win_rate": round(wins / total * 100, 1) if total else 0,
        "avg_roi": float(row[4]),
        "avg_win_roi": float(row[5]),
        "avg_loss_roi": float(row[6]),
        "best_roi": float(row[7]),
        "total_roi": float(row[8]),
        "by_type": by_type,
        "by_direction": by_dir,
    })


@router.get("/tracker", response_class=HTMLResponse)
async def trade_tracker_page():
    import os
    template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates", "tracker.html")
    with open(template_path, "r") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


_TRACKER_HTML_OLD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Trade Tracker - Crypto Perps Signals</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0a0e17;color:#e1e5ee;min-height:100vh}
.header{background:linear-gradient(135deg,#0f1629 0%,#1a1f3a 100%);padding:24px 32px;border-bottom:1px solid #1e2642}
.header h1{font-size:24px;font-weight:700;color:#fff}
.header h1 span{color:#00d4aa}
.header p{color:#7a82a6;font-size:14px;margin-top:4px}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;padding:20px 32px}
.stat-card{background:#111827;border:1px solid #1e2642;border-radius:10px;padding:16px;text-align:center}
.stat-card .label{font-size:11px;color:#7a82a6;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px}
.stat-card .value{font-size:22px;font-weight:700}
.stat-card .value.green{color:#00d4aa}
.stat-card .value.red{color:#ff4757}
.stat-card .value.blue{color:#5b8def}
.stat-card .value.yellow{color:#ffa502}
.controls{padding:16px 32px;display:flex;flex-wrap:wrap;gap:10px;align-items:center;border-bottom:1px solid #1e2642}
.controls select,.controls input{background:#111827;border:1px solid #2a3154;color:#e1e5ee;padding:8px 12px;border-radius:6px;font-size:13px;outline:none}
.controls select:focus,.controls input:focus{border-color:#5b8def}
.controls label{font-size:12px;color:#7a82a6;margin-right:4px}
.filter-group{display:flex;align-items:center;gap:4px}
.table-wrap{padding:0 32px 32px;overflow-x:auto}
table{width:100%;border-collapse:collapse;margin-top:16px;font-size:13px}
thead th{background:#111827;color:#7a82a6;font-weight:600;text-transform:uppercase;font-size:11px;letter-spacing:0.5px;padding:10px 12px;text-align:left;border-bottom:2px solid #1e2642;cursor:pointer;white-space:nowrap;user-select:none}
thead th:hover{color:#5b8def}
thead th.sorted-asc::after{content:" ▲";color:#5b8def}
thead th.sorted-desc::after{content:" ▼";color:#5b8def}
tbody tr{border-bottom:1px solid #131b2e;transition:background 0.15s}
tbody tr:hover{background:#131b2e}
td{padding:10px 12px;white-space:nowrap}
.sym{font-weight:700;color:#fff}
.dir-long{color:#00d4aa;font-weight:600}
.dir-short{color:#ff6b81;font-weight:600}
.win{color:#00d4aa;font-weight:700}
.loss{color:#ff4757;font-weight:700}
.be{color:#7a82a6}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600}
.badge-win{background:rgba(0,212,170,0.15);color:#00d4aa}
.badge-loss{background:rgba(255,71,87,0.15);color:#ff4757}
.badge-be{background:rgba(122,130,166,0.15);color:#7a82a6}
.badge-open{background:rgba(91,141,239,0.15);color:#5b8def}
.type-badge{display:inline-block;padding:2px 6px;border-radius:3px;font-size:10px;font-weight:600;background:rgba(91,141,239,0.12);color:#5b8def}
.paging{display:flex;justify-content:center;align-items:center;gap:12px;padding:20px 32px}
.paging button{background:#111827;border:1px solid #2a3154;color:#e1e5ee;padding:8px 16px;border-radius:6px;cursor:pointer;font-size:13px}
.paging button:hover{background:#1a2340;border-color:#5b8def}
.paging button:disabled{opacity:0.4;cursor:not-allowed}
.paging span{color:#7a82a6;font-size:13px}
.tp-hit{color:#00d4aa}
.tp-miss{color:#2a3154}
.breakdown{padding:0 32px 20px;display:flex;flex-wrap:wrap;gap:12px}
.breakdown-card{background:#111827;border:1px solid #1e2642;border-radius:8px;padding:12px 16px;min-width:180px}
.breakdown-card h3{font-size:12px;color:#5b8def;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px}
.breakdown-card .row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;color:#9ca3c0}
.breakdown-card .row .val{font-weight:600;color:#e1e5ee}
.loading{text-align:center;padding:60px;color:#7a82a6;font-size:16px}
@media(max-width:768px){
  .stats-grid{grid-template-columns:repeat(2,1fr);padding:12px 16px}
  .controls{padding:12px 16px}
  .table-wrap{padding:0 16px 16px}
  .header{padding:16px}
  td,th{padding:8px 6px;font-size:12px}
}
</style>
</head>
<body>

<div class="header">
  <h1>Trade <span>Tracker</span></h1>
  <p>All trades with ROI % - updated in real time</p>
</div>

<div id="stats" class="stats-grid"></div>
<div id="breakdown" class="breakdown"></div>

<div class="controls">
  <div class="filter-group">
    <label>Status:</label>
    <select id="f-status" onchange="loadTrades()">
      <option value="all" selected>All</option>
      <option value="open">Open</option>
      <option value="closed">Closed</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Direction:</label>
    <select id="f-dir" onchange="loadTrades()">
      <option value="all">All</option>
      <option value="LONG">Long</option>
      <option value="SHORT">Short</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Type:</label>
    <select id="f-type" onchange="loadTrades()">
      <option value="all">All</option>
      <option value="SOCIAL_SIGNAL">Social</option>
      <option value="NEWS_SIGNAL">News</option>
      <option value="TOP_GAINER">Top Gainer</option>
      <option value="MOMENTUM_RUNNER">Momentum</option>
      <option value="EARLY_MOVER">Early Mover</option>
      <option value="STANDARD">Standard</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Period:</label>
    <select id="f-days" onchange="loadTrades();loadStats()">
      <option value="7" selected>This Week</option>
      <option value="1">Today</option>
      <option value="30">30 Days</option>
      <option value="90">90 Days</option>
      <option value="">Since Upgrade</option>
    </select>
  </div>
  <div class="filter-group">
    <label>Symbol:</label>
    <input id="f-sym" type="text" placeholder="e.g. BTC" oninput="debounceLoad()" style="width:100px">
  </div>
  <div class="filter-group">
    <label>Per page:</label>
    <select id="f-perpage" onchange="loadTrades()">
      <option value="50">50</option>
      <option value="100">100</option>
      <option value="200">200</option>
    </select>
  </div>
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th data-col="opened_at" class="sorted-desc">Date</th>
        <th data-col="symbol">Symbol</th>
        <th data-col="direction">Dir</th>
        <th>Type</th>
        <th>Lev</th>
        <th data-col="entry_price">Entry</th>
        <th data-col="exit_price">Exit</th>
        <th>SL</th>
        <th>TP Targets</th>
        <th data-col="pnl_percent">ROI (%)</th>
        <th>Result</th>
        <th>Duration</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<div class="paging" id="paging"></div>

<script>
let currentPage=1,sortBy="opened_at",sortDir="desc",debounceTimer=null;

function debounceLoad(){clearTimeout(debounceTimer);debounceTimer=setTimeout(()=>{currentPage=1;loadTrades()},300)}

function fmtPrice(p){if(!p)return"-";if(p>=1000)return p.toLocaleString("en",{minimumFractionDigits:2,maximumFractionDigits:2});if(p>=1)return p.toFixed(4);if(p>=0.01)return p.toFixed(6);return p.toFixed(8)}

function fmtDate(iso){if(!iso)return"-";const d=new Date(iso+"Z");return d.toLocaleDateString("en",{month:"short",day:"numeric"})+' '+d.toLocaleTimeString("en",{hour:"2-digit",minute:"2-digit",hour12:false})}

function fmtDuration(mins){if(!mins&&mins!==0)return"-";if(mins<60)return mins.toFixed(0)+"m";if(mins<1440)return(mins/60).toFixed(1)+"h";return(mins/1440).toFixed(1)+"d"}

async function loadStats(){
  const days=document.getElementById("f-days").value;
  const url="/api/trades/stats"+(days?"?days="+days:"");
  try{
    const r=await fetch(url);const d=await r.json();
    document.getElementById("stats").innerHTML=`
      <div class="stat-card"><div class="label">Total Trades</div><div class="value blue">${d.total.toLocaleString()}</div></div>
      <div class="stat-card"><div class="label">Win Rate</div><div class="value ${d.win_rate>=50?'green':'red'}">${d.win_rate}%</div></div>
      <div class="stat-card"><div class="label">Wins / Losses</div><div class="value"><span class="green">${d.wins}</span> / <span class="red">${d.losses}</span></div></div>
      <div class="stat-card"><div class="label">Avg ROI</div><div class="value ${d.avg_roi>=0?'green':'red'}">${d.avg_roi}%</div></div>
      <div class="stat-card"><div class="label">Avg Win ROI</div><div class="value green">${d.avg_win_roi}%</div></div>
      <div class="stat-card"><div class="label">Avg Loss ROI</div><div class="value red">${d.avg_loss_roi}%</div></div>
      <div class="stat-card"><div class="label">Best Trade ROI</div><div class="value green">${d.best_roi}%</div></div>
      <div class="stat-card"><div class="label">Total ROI</div><div class="value ${d.total_roi>=0?'green':'red'}">${d.total_roi.toLocaleString("en",{minimumFractionDigits:2})}%</div></div>
    `;

    let bhtml="";
    if(Object.keys(d.by_type).length){
      bhtml+='<div class="breakdown-card"><h3>By Signal Type</h3>';
      for(const[k,v]of Object.entries(d.by_type)){
        bhtml+=`<div class="row"><span>${k}</span><span class="val">${v.count} trades | ${v.win_rate}% WR | <span style="color:${v.avg_roi>=0?'#00d4aa':'#ff4757'}">${v.avg_roi}% avg</span></span></div>`;
      }
      bhtml+="</div>";
    }
    if(Object.keys(d.by_direction).length){
      bhtml+='<div class="breakdown-card"><h3>By Direction</h3>';
      for(const[k,v]of Object.entries(d.by_direction)){
        bhtml+=`<div class="row"><span>${k}</span><span class="val">${v.count} trades | ${v.win_rate}% WR | <span style="color:${v.avg_roi>=0?'#00d4aa':'#ff4757'}">${v.avg_roi}% avg</span></span></div>`;
      }
      bhtml+="</div>";
    }
    document.getElementById("breakdown").innerHTML=bhtml;
  }catch(e){console.error(e)}
}

async function loadTrades(){
  const status=document.getElementById("f-status").value;
  const dir=document.getElementById("f-dir").value;
  const type=document.getElementById("f-type").value;
  const days=document.getElementById("f-days").value;
  const sym=document.getElementById("f-sym").value.trim();
  const pp=document.getElementById("f-perpage").value;

  let url=`/api/trades?page=${currentPage}&per_page=${pp}&sort_by=${sortBy}&sort_dir=${sortDir}`;
  if(status!=="all")url+=`&status=${status}`;
  if(dir!=="all")url+=`&direction=${dir}`;
  if(type!=="all")url+=`&trade_type=${type}`;
  if(days)url+=`&days=${days}`;
  if(sym)url+=`&symbol=${encodeURIComponent(sym)}`;

  const tbody=document.getElementById("tbody");
  tbody.innerHTML='<tr><td colspan="12" class="loading">Loading...</td></tr>';

  try{
    const r=await fetch(url);const d=await r.json();
    if(!d.trades.length){tbody.innerHTML='<tr><td colspan="12" style="text-align:center;padding:40px;color:#7a82a6">No trades found</td></tr>';updatePaging(d);return}

    let html="";
    for(const t of d.trades){
      const dirCls=t.direction==="LONG"?"dir-long":"dir-short";
      const resCls=t.result==="WIN"||t.result==="RUNNING +"?"badge-win":t.result==="LOSS"||t.result==="RUNNING -"?"badge-loss":"badge-be";
      const pnlCls=t.pnl>0?"win":t.pnl<0?"loss":"be";
      const roiCls=t.pnl_percent>0?"win":t.pnl_percent<0?"loss":"be";

      const tp1=t.tp1?fmtPrice(t.tp1):"-";
      const tp2=t.tp2?fmtPrice(t.tp2):"-";
      const tp3=t.tp3?fmtPrice(t.tp3):"-";
      const h1=t.tp1_hit?'tp-hit':'tp-miss';
      const h2=t.tp2_hit?'tp-hit':'tp-miss';
      const h3=t.tp3_hit?'tp-hit':'tp-miss';

      let statusBadge;
      if(t.status==="open"){
        if(t.current_price){
          const runCls=t.pnl_percent>=0?"badge-win":"badge-loss";
          statusBadge=`<span class="badge ${runCls}">${t.result} ${t.pnl_percent.toFixed(1)}%</span>`;
        } else {
          statusBadge='<span class="badge badge-open">OPEN</span>';
        }
      } else {
        statusBadge=`<span class="badge ${resCls}">${t.result}</span>`;
      }

      const exitCol=t.status==="open"&&t.current_price?`<span style="color:#ffa502">${fmtPrice(t.current_price)}</span>`:t.exit_price?fmtPrice(t.exit_price):"-";

      html+=`<tr>
        <td>${fmtDate(t.opened_at)}</td>
        <td class="sym">${t.symbol}</td>
        <td class="${dirCls}">${t.direction}</td>
        <td><span class="type-badge">${t.trade_type}</span></td>
        <td>${t.leverage ? t.leverage+'x' : '-'}</td>
        <td>${fmtPrice(t.entry_price)}</td>
        <td>${exitCol}</td>
        <td>${fmtPrice(t.stop_loss)}</td>
        <td><span class="${h1}">${tp1}</span> / <span class="${h2}">${tp2}</span> / <span class="${h3}">${tp3}</span></td>
        <td class="${roiCls}">${t.pnl_percent.toFixed(2)}%</td>
        <td>${statusBadge}</td>
        <td>${t.status==="open"?"LIVE":fmtDuration(t.duration_mins)}</td>
      </tr>`;
    }
    tbody.innerHTML=html;
    updatePaging(d);
  }catch(e){tbody.innerHTML=`<tr><td colspan="12" style="text-align:center;padding:40px;color:#ff4757">Error loading trades</td></tr>`;console.error(e)}
}

function updatePaging(d){
  const pg=document.getElementById("paging");
  pg.innerHTML=`
    <button onclick="goPage(1)" ${d.page<=1?'disabled':''}>First</button>
    <button onclick="goPage(${d.page-1})" ${d.page<=1?'disabled':''}>Prev</button>
    <span>Page ${d.page} of ${d.total_pages} (${d.total.toLocaleString()} trades)</span>
    <button onclick="goPage(${d.page+1})" ${d.page>=d.total_pages?'disabled':''}>Next</button>
    <button onclick="goPage(${d.total_pages})" ${d.page>=d.total_pages?'disabled':''}>Last</button>
  `;
}

function goPage(p){currentPage=p;loadTrades()}

document.querySelectorAll("thead th[data-col]").forEach(th=>{
  th.addEventListener("click",()=>{
    const col=th.dataset.col;
    if(sortBy===col){sortDir=sortDir==="desc"?"asc":"desc"}else{sortBy=col;sortDir="desc"}
    document.querySelectorAll("thead th").forEach(h=>{h.classList.remove("sorted-asc","sorted-desc")});
    th.classList.add(sortDir==="asc"?"sorted-asc":"sorted-desc");
    currentPage=1;loadTrades();
  });
});

loadStats();loadTrades();

setInterval(()=>{
  const status=document.getElementById("f-status").value;
  if(status==="open"||status==="all"){loadTrades()}
},15000);
</script>
</body>
</html>"""
