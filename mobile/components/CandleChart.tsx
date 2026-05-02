import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import Svg, { Line, Rect, Circle, G } from 'react-native-svg';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { runOnJS, useSharedValue } from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';

const MIN_VISIBLE_CANDLES = 12;

export type Candle = {
  time: number;     // unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
};

export type ChartMarker = {
  time: number;       // unix seconds — placed at the nearest candle
  price: number;
  kind: 'open' | 'close-win' | 'close-loss';
  direction?: 'LONG' | 'SHORT';
};

export type ChartZone = {
  /** Start of the zone in unix seconds (formation/origin candle). */
  fromTime: number;
  /** End of the zone in unix seconds. If omitted, extends to the right edge. */
  toTime?: number;
  /** Top price of the zone. */
  top: number;
  /** Bottom price of the zone. */
  bottom: number;
  /** Bull = green-tinted (support); bear = red-tinted (resistance). */
  side: 'bull' | 'bear';
  /** Optional dim styling for filled/used zones. */
  dim?: boolean;
};

export type ChartPriceLine = {
  /** Y-axis price level. */
  price: number;
  /** 'buy' = green wall (bid), 'sell' = red wall (ask). */
  side: 'buy' | 'sell';
  /** Short label rendered at the right edge (e.g. "$2.4M"). */
  label?: string;
};

/**
 * Compact native candlestick chart. Renders the most recent N candles with
 * optional entry/exit markers overlayed at their (time, price) coordinates.
 *
 * Designed for the strategy detail screen — fixed height, fills width.
 */
export function CandleChart({
  candles,
  markers = [],
  zones = [],
  priceLines = [],
  width,
  height = 180,
  symbol,
  tf,
  showOhlcLegend = false,
  livePrice,
  interactive = true,
}: {
  candles: Candle[];
  markers?: ChartMarker[];
  zones?: ChartZone[];
  priceLines?: ChartPriceLine[];
  width: number;
  height?: number;
  symbol?: string;
  tf?: string;
  /** When true, renders an OHLC info strip overlay in the top-left of the chart. */
  showOhlcLegend?: boolean;
  /** When provided, draws a pulsing live-price marker at this y-level. */
  livePrice?: number;
  /** When true (default), enables pinch-to-zoom and horizontal pan. */
  interactive?: boolean;
}) {
  const padX = 8;
  const padY = 14;
  const innerW = Math.max(width - padX * 2, 1);
  const innerH = Math.max(height - padY * 2, 1);

  // ─── Viewport (pinch-zoom + horizontal pan) ─────────────────────────────
  // Internally we keep the visible window as (start, count) into the full
  // candles array. Gestures update these via shared values during the gesture
  // and commit to React state for redraw.
  const total = candles?.length || 0;
  const [viewCount, setViewCount] = useState<number>(total);
  const [viewStart, setViewStart] = useState<number>(0);

  // Whenever a new symbol / timeframe arrives the upstream candles array's
  // first-candle time changes — reset the viewport to "show everything".
  const firstTime = candles?.[0]?.time;
  const lastResetKey = useRef<string>('');
  useEffect(() => {
    const key = `${firstTime}|${total}|${tf || ''}|${symbol || ''}`;
    // Only reset on symbol/tf change, NOT on every new candle tick.
    const baseKey = `${tf || ''}|${symbol || ''}`;
    const lastBase = lastResetKey.current.split('::')[0];
    if (baseKey !== lastBase) {
      setViewStart(0);
      setViewCount(total);
      lastResetKey.current = `${baseKey}::${key}`;
    } else if (viewCount === 0 || viewCount > total) {
      // First-load case where total was 0 then jumped.
      setViewStart(Math.max(0, total - viewCount || 0));
      setViewCount(total);
      lastResetKey.current = `${baseKey}::${key}`;
    }
  }, [firstTime, total, tf, symbol, viewCount]);

  const savedCount = useSharedValue(viewCount);
  const savedStart = useSharedValue(viewStart);

  const commitView = (start: number, count: number) => {
    setViewStart(start);
    setViewCount(count);
  };

  const pinch = Gesture.Pinch()
    .onStart(() => {
      savedCount.value = viewCount;
      savedStart.value = viewStart;
    })
    .onUpdate((e) => {
      if (!interactive || total === 0) return;
      const nextCount = Math.max(
        MIN_VISIBLE_CANDLES,
        Math.min(total, Math.round(savedCount.value / Math.max(0.1, e.scale))),
      );
      // Anchor the zoom around the right edge so it feels like price-action zoom.
      const rightEdge = savedStart.value + savedCount.value;
      const nextStart = Math.max(0, Math.min(total - nextCount, rightEdge - nextCount));
      runOnJS(commitView)(nextStart, nextCount);
    });

  const pan = Gesture.Pan()
    .minDistance(6)
    .onStart(() => {
      savedStart.value = viewStart;
      savedCount.value = viewCount;
    })
    .onUpdate((e) => {
      if (!interactive || total === 0) return;
      const candleW = innerW / Math.max(savedCount.value, 1);
      const dCandle = -e.translationX / Math.max(candleW, 0.1);
      const nextStart = Math.max(
        0,
        Math.min(total - savedCount.value, Math.round(savedStart.value + dCandle)),
      );
      if (nextStart !== viewStart) runOnJS(setViewStart)(nextStart);
    });

  const gesture = Gesture.Simultaneous(pinch, pan);

  // Effective slice we actually render.
  const visibleCandles = useMemo(() => {
    if (total === 0) return [];
    const start = Math.max(0, Math.min(total - 1, viewStart));
    const count = Math.max(MIN_VISIBLE_CANDLES, Math.min(total - start, viewCount || total));
    return candles.slice(start, start + count);
  }, [candles, viewStart, viewCount, total]);

  const isZoomed = viewCount > 0 && viewCount < total;

  const onResetZoom = () => {
    setViewStart(0);
    setViewCount(total);
  };

  const { paths, yMin, yMax, xForTime, yForPrice } = useMemo(() => {
    if (!visibleCandles || visibleCandles.length === 0) {
      return { paths: [] as any[], yMin: 0, yMax: 1, xForTime: () => 0, yForPrice: () => 0 };
    }
    // Use the visible window for axis scaling so zoom feels useful.
    const candles = visibleCandles;
    const lows = candles.map((c) => c.low);
    const highs = candles.map((c) => c.high);
    let lo = Math.min(...lows);
    let hi = Math.max(...highs);
    // pad the y-range so candles don't touch top/bottom edges
    const pad = (hi - lo) * 0.05 || hi * 0.001 || 1;
    lo -= pad; hi += pad;
    const range = (hi - lo) || 1;

    const slotW = innerW / candles.length;
    const bodyW = Math.max(slotW * 0.6, 1.5);

    const xForIdx = (i: number) => padX + slotW * (i + 0.5);
    const yFor = (p: number) => padY + innerH - ((p - lo) / range) * innerH;
    const timeMin = candles[0].time;
    const timeMax = candles[candles.length - 1].time;
    const timeSpan = (timeMax - timeMin) || 1;
    const xForT = (t: number) => {
      const clamped = Math.max(timeMin, Math.min(timeMax, t));
      return padX + ((clamped - timeMin) / timeSpan) * innerW;
    };

    const out = candles.map((c, i) => {
      const x = xForIdx(i);
      const yHigh = yFor(c.high);
      const yLow = yFor(c.low);
      const yOpen = yFor(c.open);
      const yClose = yFor(c.close);
      const up = c.close >= c.open;
      const top = Math.min(yOpen, yClose);
      const bodyH = Math.max(Math.abs(yOpen - yClose), 1);
      return {
        x, yHigh, yLow, top, bodyH, bodyW,
        color: up ? colors.positive : colors.negative,
      };
    });

    return {
      paths: out,
      yMin: lo,
      yMax: hi,
      xForTime: xForT,
      yForPrice: yFor,
    };
  }, [candles, innerH, innerW, padX, padY]);

  if (total === 0 || visibleCandles.length === 0) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>No price data available.</Text>
      </View>
    );
  }

  const lastClose = visibleCandles[visibleCandles.length - 1].close;
  const firstOpen = visibleCandles[0].open;
  const change = lastClose - firstOpen;
  const changePct = (change / firstOpen) * 100;
  const liveY = livePrice != null ? yForPrice(livePrice) : null;

  return (
    <View style={styles.wrap}>
      {(symbol || tf) ? (
        <View style={styles.legend}>
          <Text style={styles.legendSym}>
            {symbol?.replace('USDT', '')}{tf ? `  ·  ${tf}` : ''}
          </Text>
          <Text style={[
            styles.legendChange,
            { color: change >= 0 ? colors.positive : colors.negative },
          ]}>
            {change >= 0 ? '+' : ''}{changePct.toFixed(2)}%
          </Text>
        </View>
      ) : null}
      <GestureDetector gesture={gesture}>
      <Svg width={width} height={height}>
        {/* horizontal grid (faint) */}
        {[0.25, 0.5, 0.75].map((f) => {
          const y = padY + innerH * f;
          return (
            <Line
              key={`g-${f}`}
              x1={padX} y1={y} x2={width - padX} y2={y}
              stroke={colors.border} strokeWidth={1} strokeDasharray="3,5" opacity={0.5}
            />
          );
        })}

        {/* FVG / liquidity zones — drawn UNDER candles so wicks stay visible */}
        {zones.map((z, i) => {
          const x1 = xForTime(z.fromTime);
          const x2 = z.toTime ? xForTime(z.toTime) : (width - padX);
          const yTop = yForPrice(z.top);
          const yBot = yForPrice(z.bottom);
          const x = Math.min(x1, x2);
          const w = Math.max(Math.abs(x2 - x1), 2);
          const y = Math.min(yTop, yBot);
          const h = Math.max(Math.abs(yBot - yTop), 1);
          const baseFill = z.side === 'bull' ? colors.positive : colors.negative;
          return (
            <G key={`z-${i}`}>
              <Rect
                x={x} y={y} width={w} height={h}
                fill={baseFill}
                opacity={z.dim ? 0.10 : 0.22}
              />
              <Line
                x1={x} y1={yTop} x2={x + w} y2={yTop}
                stroke={baseFill} strokeWidth={1} opacity={z.dim ? 0.4 : 0.85}
                strokeDasharray="2,3"
              />
              <Line
                x1={x} y1={yBot} x2={x + w} y2={yBot}
                stroke={baseFill} strokeWidth={1} opacity={z.dim ? 0.4 : 0.85}
                strokeDasharray="2,3"
              />
            </G>
          );
        })}

        {/* horizontal price lines (order-book walls) — drawn UNDER candles */}
        {priceLines.map((pl, i) => {
          const y = yForPrice(pl.price);
          if (!Number.isFinite(y) || y < padY || y > padY + innerH) return null;
          const stroke = pl.side === 'buy' ? colors.positive : colors.negative;
          return (
            <Line
              key={`pl-${i}`}
              x1={padX} y1={y} x2={width - padX} y2={y}
              stroke={stroke} strokeWidth={1.5} opacity={0.75}
              strokeDasharray="6,4"
            />
          );
        })}

        {/* candles */}
        {paths.map((p, i) => (
          <G key={`c-${i}`}>
            <Line
              x1={p.x} y1={p.yHigh} x2={p.x} y2={p.yLow}
              stroke={p.color} strokeWidth={1}
            />
            <Rect
              x={p.x - p.bodyW / 2}
              y={p.top}
              width={p.bodyW}
              height={p.bodyH}
              fill={p.color}
              rx={0.5}
            />
          </G>
        ))}

        {/* markers (entry/exit) */}
        {markers.map((m, i) => {
          const x = xForTime(m.time);
          const y = yForPrice(m.price);
          if (m.kind === 'open') {
            const fill = m.direction === 'SHORT' ? colors.negative : colors.accent;
            return (
              <G key={`m-${i}`}>
                <Circle cx={x} cy={y} r={4.5} fill={fill} stroke={colors.bg} strokeWidth={1.5} />
              </G>
            );
          }
          // close marker — small diamond
          const c = m.kind === 'close-win' ? colors.positive : colors.negative;
          const s = 4;
          return (
            <G key={`m-${i}`}>
              <Rect
                x={x - s} y={y - s} width={s * 2} height={s * 2}
                fill={c} stroke={colors.bg} strokeWidth={1}
                transform={`rotate(45 ${x} ${y})`}
              />
            </G>
          );
        })}
        {/* Live-price marker — pulsing dot at the right edge so the user
            can see the chart "ticking" by the millisecond. */}
        {liveY != null && Number.isFinite(liveY) && liveY >= padY && liveY <= padY + innerH ? (
          <G>
            <Line
              x1={padX} y1={liveY} x2={width - padX - 4} y2={liveY}
              stroke={colors.accent} strokeWidth={1} opacity={0.5}
              strokeDasharray="2,3"
            />
            <Circle
              cx={width - padX - 2} cy={liveY} r={5}
              fill={colors.accent} opacity={0.25}
            />
            <Circle
              cx={width - padX - 2} cy={liveY} r={2.6}
              fill={colors.accent}
            />
          </G>
        ) : null}
      </Svg>
      </GestureDetector>
      {isZoomed ? (
        <Pressable onPress={onResetZoom} style={styles.resetBtn} hitSlop={6}>
          <Ionicons name="contract" size={11} color={colors.accent} />
          <Text style={styles.resetBtnText}>FIT</Text>
        </Pressable>
      ) : null}
      {showOhlcLegend ? (
        <View style={styles.ohlcOverlay} pointerEvents="none">
          {(() => {
            const c = visibleCandles[visibleCandles.length - 1];
            const up = c.close >= c.open;
            const tone = up ? colors.positive : colors.negative;
            const fmt = (n: number) =>
              n >= 1000 ? n.toLocaleString(undefined, { maximumFractionDigits: 2 })
                        : n.toFixed(n >= 1 ? 3 : n >= 0.01 ? 5 : 8);
            return (
              <Text style={styles.ohlcText}>
                <Text style={{ color: colors.textMute }}>O </Text>
                <Text style={{ color: tone }}>{fmt(c.open)}  </Text>
                <Text style={{ color: colors.textMute }}>H </Text>
                <Text style={{ color: tone }}>{fmt(c.high)}  </Text>
                <Text style={{ color: colors.textMute }}>L </Text>
                <Text style={{ color: tone }}>{fmt(c.low)}  </Text>
                <Text style={{ color: colors.textMute }}>C </Text>
                <Text style={{ color: tone }}>{fmt(c.close)}</Text>
              </Text>
            );
          })()}
        </View>
      ) : null}
      {/* Right-edge wall labels — overlaid as absolute text so they sit
          flush with the chart's right gutter without distorting the SVG. */}
      {priceLines.length > 0 ? (
        <View style={styles.wallLabelOverlay} pointerEvents="none">
          {priceLines.map((pl, i) => {
            const y = yForPrice(pl.price);
            if (!Number.isFinite(y) || y < padY || y > padY + innerH) return null;
            const tone = pl.side === 'buy' ? colors.positive : colors.negative;
            return (
              <Text
                key={`pll-${i}`}
                style={[
                  styles.wallLabel,
                  { top: y - 7, color: tone, borderColor: tone },
                ]}
              >
                {pl.label || `$${pl.price.toFixed(pl.price >= 100 ? 0 : 2)}`}
              </Text>
            );
          })}
        </View>
      ) : null}
      <View style={styles.priceRow}>
        <Text style={styles.priceLabel}>Last</Text>
        <Text style={styles.priceVal}>${lastClose.toLocaleString(undefined, { maximumFractionDigits: 4 })}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.sm,
    overflow: 'hidden',
  },
  legend: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    marginBottom: 4,
  },
  legendSym: { color: colors.text, fontFamily: font.bold, fontSize: 12, letterSpacing: 0.5 },
  legendChange: { fontFamily: font.bold, fontSize: 12, fontVariant: ['tabular-nums'] },
  priceRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 4,
    marginTop: 4,
  },
  priceLabel: { color: colors.textMute, fontFamily: font.semibold, fontSize: 11, letterSpacing: 0.5, textTransform: 'uppercase' },
  priceVal: { color: colors.text, fontFamily: font.bold, fontSize: 13, fontVariant: ['tabular-nums'] },
  empty: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.lg,
  },
  emptyText: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 13,
    textAlign: 'center',
  },
  ohlcOverlay: {
    position: 'absolute',
    top: 30,
    left: 12,
    paddingHorizontal: 6,
    paddingVertical: 3,
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderRadius: 4,
  },
  ohlcText: {
    fontFamily: font.semibold,
    fontSize: 10,
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.2,
  },
  resetBtn: {
    position: 'absolute',
    top: 30,
    right: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(34,211,238,0.15)',
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.45)',
    borderRadius: radius.pill,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  resetBtnText: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.6,
  },
  wallLabelOverlay: {
    position: 'absolute',
    right: 4,
    top: 24,            // accounts for the legend strip above the SVG
    bottom: 22,         // accounts for the priceRow strip below the SVG
    width: 60,
  },
  wallLabel: {
    position: 'absolute',
    right: 0,
    fontFamily: font.bold,
    fontSize: 9,
    fontVariant: ['tabular-nums'],
    paddingHorizontal: 4,
    paddingVertical: 1,
    borderRadius: 3,
    borderWidth: 1,
    backgroundColor: 'rgba(0,0,0,0.55)',
    overflow: 'hidden',
  },
});
