import React, { useEffect, useMemo, useRef } from 'react';
import { View, StyleSheet, ActivityIndicator, Text } from 'react-native';
import { WebView } from 'react-native-webview';
import { colors, font } from '@/constants/colors';
// Relative path (not the @/ alias) — Expo's Metro asset resolver excludes
// the `assets/` tree from module resolution, so the alias would fail here.
import { LIGHTWEIGHT_CHARTS_SRC } from '../assets/vendor/lightweightCharts';

/**
 * TradingView-quality candlestick chart for the mobile Trade tab.
 *
 * Embeds the **same** TradingView Lightweight Charts library the web /trade
 * page uses (loaded from unpkg) inside a WebView, so we get pixel-identical
 * candles, volume histogram, OHLC legend, native crosshair, smooth pinch/pan
 * zoom, and price-axis live-value tag — all rendered on a real Canvas, not
 * SVG. Wall lines and FVG zones are drawn via Lightweight Charts' native
 * `createPriceLine()` API so they always sit at the right Y-coordinate even
 * after the user pans/zooms.
 *
 * The WebView is rendered ONCE with the HTML template; subsequent prop
 * changes are pushed in via `postMessage` as a JSON payload that the page's
 * `update()` function consumes. This avoids any reflow / re-init when the
 * 1Hz ticker price ticks — the live tag just slides smoothly.
 */

export type WCandle = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type WWall = {
  price: number;
  side: 'buy' | 'sell';
  size_usd: number;
};

export type WZone = {
  fromTime: number;
  top: number;
  bottom: number;
  side: 'bull' | 'bear';
  filled?: boolean;
};

export function TradeChartWebView({
  candles,
  walls,
  zones,
  livePrice,
  symbol,
  height = 380,
  showWalls = true,
  showFvg = true,
  decimals,
}: {
  candles: WCandle[];
  walls?: WWall[];
  zones?: WZone[];
  livePrice?: number | null;
  symbol: string;
  height?: number;
  showWalls?: boolean;
  showFvg?: boolean;
  /** Decimal places for price formatting; defaults to a sensible auto-pick. */
  decimals?: number;
}) {
  const webRef = useRef<WebView>(null);
  const readyRef = useRef(false);
  // Pending payload buffered until the WebView's HTML signals it's ready.
  const pendingRef = useRef<string | null>(null);

  // Pick a sensible price decimal precision based on the live or last close
  // price. Matches the web /trade SYMBOL_META table closely enough for charts.
  const dec = useMemo(() => {
    if (typeof decimals === 'number') return decimals;
    const p = livePrice || (candles.length ? candles[candles.length - 1].close : 0);
    if (p >= 1000) return 2;
    if (p >= 1)    return 4;
    if (p >= 0.01) return 6;
    return 8;
  }, [decimals, livePrice, candles]);

  // Build the full payload the WebView's update() expects.
  const payload = useMemo(
    () =>
      JSON.stringify({
        candles: candles.map((c) => ({
          time: c.time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
          volume: c.volume || 0,
        })),
        walls: showWalls ? (walls || []) : [],
        zones: showFvg ? (zones || []).filter((z) => !z.filled) : [],
        livePrice: livePrice || null,
        symbol,
        dec,
      }),
    [candles, walls, zones, livePrice, symbol, dec, showWalls, showFvg],
  );

  // Push the payload to the WebView whenever it changes. If the WebView hasn't
  // signalled "ready" yet, buffer the latest payload and flush on ready.
  useEffect(() => {
    if (!webRef.current) return;
    if (!readyRef.current) {
      pendingRef.current = payload;
      return;
    }
    // injectJavaScript is more reliable than postMessage on Android <8.
    const js = `window.__chartUpdate && window.__chartUpdate(${payload}); true;`;
    webRef.current.injectJavaScript(js);
  }, [payload]);

  return (
    // Claim the responder before the parent ScrollView can — without this
    // the outer Screen ScrollView swallows pinch/pan touches before the
    // WebView's TradingView library ever sees them, so the chart appears
    // "uncontrollable". onMoveShouldSetResponderCapture wins arbitration
    // against any ancestor scroll view, and onResponderTerminationRequest
    // false prevents iOS from yanking the touch away mid-gesture.
    <View
      style={[styles.wrap, { height }]}
      onStartShouldSetResponder={() => true}
      onMoveShouldSetResponderCapture={() => true}
      onResponderTerminationRequest={() => false}
    >
      <WebView
        ref={webRef}
        originWhitelist={['*']}
        source={{ html: HTML_TEMPLATE.replace('__LWC_SRC__', LIGHTWEIGHT_CHARTS_SRC) }}
        // Performance + UX flags
        javaScriptEnabled
        domStorageEnabled
        allowsInlineMediaPlayback
        mediaPlaybackRequiresUserAction={false}
        scrollEnabled={false}
        overScrollMode="never"
        bounces={false}
        // Prevent Android from delegating touches up to the parent ScrollView
        // — the WebView (and the TradingView lib inside it) must own them.
        nestedScrollEnabled={false}
        // Render a native black bg under the WebView so first-paint isn't white
        style={styles.web}
        containerStyle={styles.webContainer}
        // The HTML page posts "ready" once Lightweight Charts has booted; flush
        // any buffered payload immediately so first paint shows real candles.
        onMessage={(e) => {
          const msg = e.nativeEvent.data;
          if (msg === 'ready') {
            readyRef.current = true;
            if (pendingRef.current && webRef.current) {
              const js = `window.__chartUpdate && window.__chartUpdate(${pendingRef.current}); true;`;
              webRef.current.injectJavaScript(js);
              pendingRef.current = null;
            }
          }
        }}
        // If the WebView ever reloads / crashes / process-recovers, treat the
        // page as un-booted again so the next props change buffers and waits
        // for the new "ready" handshake instead of injecting into a dead page.
        onLoadStart={() => {
          readyRef.current = false;
          // Buffer the latest known payload so it flushes on the next ready.
          pendingRef.current = payload;
        }}
        onContentProcessDidTerminate={() => {
          readyRef.current = false;
          pendingRef.current = payload;
          webRef.current?.reload();
        }}
        // No nav inside the chart — block any external link the embedded
        // script could try to open.
        onShouldStartLoadWithRequest={(req) => req.url.startsWith('about:') || req.url.startsWith('data:')}
        renderLoading={() => (
          <View style={styles.loading}>
            <ActivityIndicator size="small" color={colors.accent} />
            <Text style={styles.loadingText}>Loading TradingView chart…</Text>
          </View>
        )}
        startInLoadingState
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    width: '100%',
    backgroundColor: '#131722',
    borderRadius: 14,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#1e222d',
  },
  web: { flex: 1, backgroundColor: '#131722' },
  webContainer: { backgroundColor: '#131722' },
  loading: {
    position: 'absolute',
    inset: 0 as unknown as number,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#131722',
  },
  loadingText: {
    color: colors.textDim,
    marginTop: 8,
    fontFamily: font.medium,
    fontSize: 12,
  },
});

// ─────────────────────────────────────────────────────────────────────────
// Embedded chart HTML. Loaded as a single static document; data is pushed in
// via `window.__chartUpdate(payload)` from the React side. Mirrors the web
// /trade page chart styling (#131722 bg, 26a69a/ef5350 candles, JetBrains
// Mono for the OHLC legend, native crosshair).
// ─────────────────────────────────────────────────────────────────────────
const HTML_TEMPLATE = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover" />
<style>
  *,*::before,*::after { box-sizing: border-box; }
  html, body { margin:0; padding:0; height:100%; background:#131722; overflow:hidden;
               font-family: -apple-system, system-ui, "Inter", sans-serif; color:#9ca3af; }
  #chart { position:absolute; inset:0; }
  /* OHLC legend overlay (top-left of chart). Matches web /trade. */
  #legend {
    position:absolute; top:8px; left:10px; z-index:5;
    display:flex; gap:10px; align-items:center; pointer-events:none;
    font: 600 11px/1.2 "JetBrains Mono", ui-monospace, Menlo, monospace;
    background: rgba(19,23,34,0.55); padding:6px 10px; border-radius:6px;
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
  }
  #legend .sym { color:#d1d4dc; font-weight:700; letter-spacing:0.4px; }
  #legend .lab { color:#787b86; margin-right:2px; }
  #legend .v   { color:#d1d4dc; }
  #legend .v.up { color:#26a69a; }
  #legend .v.dn { color:#ef5350; }
  /* Empty-state veil */
  #empty {
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    color:#787b86; font: 500 13px/1.4 -apple-system, "Inter", sans-serif;
    background:#131722; z-index:10;
  }
  #empty.hidden { display:none; }
</style>
</head>
<body>
  <div id="chart"></div>
  <div id="legend">
    <span class="sym" id="lg-sym">—</span>
    <span><span class="lab">O</span><span class="v" id="lg-o">—</span></span>
    <span><span class="lab">H</span><span class="v" id="lg-h">—</span></span>
    <span><span class="lab">L</span><span class="v" id="lg-l">—</span></span>
    <span><span class="lab">C</span><span class="v" id="lg-c">—</span></span>
    <span class="v" id="lg-chg">—</span>
  </div>
  <div id="empty">Loading chart…</div>
  <script>__LWC_SRC__</script>
  <script>
    (function(){
      var chart, candleSeries, volumeSeries;
      var fullCandles = [];
      var wallLines = [];   // priceLine handles for walls
      var zoneLines = [];   // priceLine handles for FVG top+bottom
      var dec = 2;
      var minMove = 0.01;
      var fmt = function(p){ try { return Number(p).toLocaleString(undefined,{minimumFractionDigits:dec, maximumFractionDigits:dec}); } catch(_) { return String(p); } };

      function init(){
        var el = document.getElementById('chart');
        chart = LightweightCharts.createChart(el, {
          layout: { background:{type:'solid', color:'#131722'}, textColor:'#9ca3af',
                    fontFamily:'Inter, -apple-system, system-ui, sans-serif', fontSize:11 },
          grid:   { vertLines:{color:'#1e222d'}, horzLines:{color:'#1e222d'} },
          rightPriceScale:{ borderColor:'#2a2e39', scaleMargins:{top:0.08, bottom:0.28} },
          timeScale:{ borderColor:'#2a2e39', timeVisible:true, secondsVisible:false, rightOffset:6 },
          crosshair:{
            mode:1,
            vertLine:{color:'#9ca3af', width:1, style:3, labelBackgroundColor:'#363a45'},
            horzLine:{color:'#9ca3af', width:1, style:3, labelBackgroundColor:'#363a45'},
          },
          // Allow BOTH axes to be driven by touch — vertTouchDrag was off in
          // v1.2 to let page scroll pass through, but the parent ScrollView is
          // now the responder owner (see TradeChartWebView wrapper) so the
          // chart can safely consume vertical motion too. This unlocks the
          // expected TradingView-style "drag in any direction" feel.
          handleScroll:{ mouseWheel:true, pressedMouseMove:true, horzTouchDrag:true, vertTouchDrag:true },
          handleScale: { axisPressedMouseMove:true, mouseWheel:true, pinch:true },
          autoSize:true,
        });
        candleSeries = chart.addCandlestickSeries({
          upColor:'#26a69a', downColor:'#ef5350',
          borderUpColor:'#26a69a', borderDownColor:'#ef5350',
          wickUpColor:'#26a69a', wickDownColor:'#ef5350',
          priceFormat:{type:'price', precision:2, minMove:0.01},
        });
        volumeSeries = chart.addHistogramSeries({
          priceFormat:{type:'volume'},
          priceScaleId:'volume_scale',
          lastValueVisible:false,
          priceLineVisible:false,
        });
        chart.priceScale('volume_scale').applyOptions({
          scaleMargins:{top:0.78, bottom:0},
          borderVisible:false,
        });
        chart.subscribeCrosshairMove(updateLegend);

        // Tell React side we're booted and ready to receive data.
        try {
          if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
            window.ReactNativeWebView.postMessage('ready');
          }
        } catch(_){}
      }

      function updateLegend(param){
        var c;
        if (param && param.time) {
          c = fullCandles.find(function(x){ return x.time === param.time; });
        }
        if (!c) c = fullCandles[fullCandles.length - 1];
        if (!c) return;
        var chg = c.close - c.open;
        var chgPct = c.open > 0 ? (chg / c.open) * 100 : 0;
        var up = chg >= 0;
        var cls = up ? 'up' : 'dn';
        document.getElementById('lg-o').textContent = fmt(c.open);
        document.getElementById('lg-h').textContent = fmt(c.high);
        document.getElementById('lg-l').textContent = fmt(c.low);
        var cEl = document.getElementById('lg-c');
        cEl.textContent = fmt(c.close);
        cEl.className = 'v ' + cls;
        var chgEl = document.getElementById('lg-chg');
        chgEl.textContent = (up?'+':'') + chg.toFixed(dec) + ' (' + (up?'+':'') + chgPct.toFixed(2) + '%)';
        chgEl.className = 'v ' + cls;
      }

      function clearOverlayLines(){
        try {
          wallLines.forEach(function(h){ candleSeries.removePriceLine(h); });
        } catch(_){}
        try {
          zoneLines.forEach(function(h){ candleSeries.removePriceLine(h); });
        } catch(_){}
        wallLines = [];
        zoneLines = [];
      }

      // Compact USD formatter for wall labels ($1.2M, $340K).
      function fmtUsd(n){
        n = Number(n) || 0;
        if (n >= 1e9) return '$' + (n/1e9).toFixed(2) + 'B';
        if (n >= 1e6) return '$' + (n/1e6).toFixed(2) + 'M';
        if (n >= 1e3) return '$' + (n/1e3).toFixed(0) + 'K';
        return '$' + n.toFixed(0);
      }

      // Public update function — called by React via injectJavaScript.
      window.__chartUpdate = function(data){
        if (!chart) return;
        try {
          // Decimals + price format
          if (typeof data.dec === 'number' && data.dec !== dec) {
            dec = data.dec;
            minMove = Math.pow(10, -dec);
            candleSeries.applyOptions({ priceFormat:{ type:'price', precision:dec, minMove:minMove } });
          }
          // Symbol label
          if (data.symbol) {
            document.getElementById('lg-sym').textContent = data.symbol + '/USDT';
          }

          // Candles + volume
          var cs = (data.candles || []).filter(function(c){
            return c && typeof c.time === 'number' && typeof c.close === 'number';
          });
          fullCandles = cs;
          if (cs.length === 0) {
            document.getElementById('empty').classList.remove('hidden');
            return;
          }
          document.getElementById('empty').classList.add('hidden');

          candleSeries.setData(cs);
          volumeSeries.setData(cs.map(function(c){
            return {
              time:  c.time,
              value: c.volume || 0,
              color: (c.close >= c.open) ? 'rgba(38,166,154,0.45)' : 'rgba(239,83,80,0.45)',
            };
          }));

          // Live tag — overwrite the trailing candle's close so the right-edge
          // price label ticks smoothly between candle-close events.
          if (data.livePrice && cs.length) {
            var last = cs[cs.length - 1];
            candleSeries.update({
              time:  last.time,
              open:  last.open,
              high:  Math.max(last.high, data.livePrice),
              low:   Math.min(last.low,  data.livePrice),
              close: data.livePrice,
            });
          }

          // Walls + zones — clear and re-draw via priceLine API
          clearOverlayLines();

          (data.walls || []).forEach(function(w){
            var isBuy = w.side === 'buy';
            var h = candleSeries.createPriceLine({
              price: w.price,
              color: isBuy ? 'rgba(38,166,154,0.85)' : 'rgba(239,83,80,0.85)',
              lineWidth: 2,
              lineStyle: 2,           // dashed
              axisLabelVisible: true,
              title: fmtUsd(w.size_usd),
            });
            wallLines.push(h);
          });

          (data.zones || []).forEach(function(z){
            var isBull = z.side === 'bull';
            // Bull = green (support / buy zone), Bear = red (resistance / sell zone).
            // Brighter colors + axis labels so each gap is identifiable at a glance.
            var col       = isBull ? 'rgba(38,166,154,0.85)'  : 'rgba(239,83,80,0.85)';
            var labelBg   = isBull ? 'rgba(38,166,154,0.95)'  : 'rgba(239,83,80,0.95)';
            var arrow     = isBull ? '▲' : '▼';
            var topTitle  = arrow + ' FVG ' + (isBull ? 'support' : 'resistance');
            var top = candleSeries.createPriceLine({
              price: z.top,
              color: col,
              lineWidth: 1,
              lineStyle: 2,           // dashed so it's distinct from candle wicks
              axisLabelVisible: true,
              axisLabelColor: labelBg,
              axisLabelTextColor: '#ffffff',
              title: topTitle,
            });
            var bot = candleSeries.createPriceLine({
              price: z.bottom,
              color: col,
              lineWidth: 1,
              lineStyle: 2,
              axisLabelVisible: true,
              axisLabelColor: labelBg,
              axisLabelTextColor: '#ffffff',
              title: '',              // bottom shares the band with the top label
            });
            zoneLines.push(top); zoneLines.push(bot);
          });

          updateLegend();
        } catch (err) {
          // Surface errors to the React side so they show up in dev logs.
          try {
            if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {
              window.ReactNativeWebView.postMessage('err:' + (err && err.message || String(err)));
            }
          } catch(_){}
        }
      };

      // Boot once the LWC script is loaded. The script tag is synchronous so
      // LightweightCharts is already on window by the time we run.
      if (typeof LightweightCharts !== 'undefined') {
        init();
      } else {
        // Fall back — shouldn't happen, but be safe.
        var t = setInterval(function(){
          if (typeof LightweightCharts !== 'undefined') {
            clearInterval(t);
            init();
          }
        }, 50);
      }
    })();
  </script>
</body>
</html>`;
