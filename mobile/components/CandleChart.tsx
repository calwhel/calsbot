import React, { useEffect, useMemo, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import Svg, { Line, Rect, Circle, G, Text as SvgText } from 'react-native-svg';
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
  volume?: number;
};

export type ChartMarker = {
  time: number;       // unix seconds — placed at the nearest candle
  price: number;
  kind: 'open' | 'close-win' | 'close-loss';
  direction?: 'LONG' | 'SHORT';
};

export type ChartZone = {
  fromTime: number;
  toTime?: number;
  top: number;
  bottom: number;
  side: 'bull' | 'bear';
  dim?: boolean;
};

export type ChartPriceLine = {
  price: number;
  side: 'buy' | 'sell';
  label?: string;
};

/**
 * TradingView-styled candlestick chart with right-side price axis (live tag),
 * volume bars beneath the price area, and a thin time axis. Pinch-to-zoom +
 * horizontal pan are scoped so vertical motion always passes through to the
 * parent ScrollView (no more "can't touch the chart").
 */
export function CandleChart({
  candles,
  markers = [],
  zones = [],
  priceLines = [],
  width,
  height = 280,
  symbol,
  tf,
  showOhlcLegend = false,
  livePrice,
  showVolume = true,
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
  showOhlcLegend?: boolean;
  livePrice?: number;
  showVolume?: boolean;
  interactive?: boolean;
}) {
  // ─── Layout ─────────────────────────────────────────────────────────────
  // Right gutter holds the price axis labels + live price tag.
  // Bottom holds the optional volume strip and a thin time axis.
  const padTop      = 8;
  const padLeft     = 6;
  const axisRightW  = 54;
  const volH        = showVolume ? 46 : 0;
  const volGap      = showVolume ? 4 : 0;
  const timeH       = 16;
  const innerW      = Math.max(width - padLeft - axisRightW, 1);
  const candleH     = Math.max(height - padTop - volGap - volH - timeH, 60);
  const volTop      = padTop + candleH + volGap;
  const timeTop     = volTop + volH;

  // ─── Viewport (pinch-zoom + horizontal pan) ──────────────────────────────
  const total = candles?.length || 0;
  const [viewCount, setViewCount] = useState<number>(total);
  const [viewStart, setViewStart] = useState<number>(0);

  const firstTime = candles?.[0]?.time;
  const lastResetKey = useRef<string>('');
  useEffect(() => {
    const baseKey = `${tf || ''}|${symbol || ''}`;
    const lastBase = lastResetKey.current.split('::')[0];
    if (baseKey !== lastBase) {
      // Symbol/timeframe changed → reset to "show everything".
      setViewStart(0);
      setViewCount(total);
      lastResetKey.current = `${baseKey}::${firstTime}|${total}`;
    } else if (viewCount === 0 || viewCount > total) {
      // First-load case where total was 0 then jumped.
      setViewStart(Math.max(0, total - viewCount || 0));
      setViewCount(total);
      lastResetKey.current = `${baseKey}::${firstTime}|${total}`;
    }
  }, [firstTime, total, tf, symbol, viewCount]);

  const savedCount = useSharedValue(viewCount);
  const savedStart = useSharedValue(viewStart);

  const commitView = (start: number, count: number) => {
    setViewStart(start);
    setViewCount(count);
  };

  // CRITICAL: scope pan to horizontal motion only. `activeOffsetX` makes the
  // pan claim the touch only after >8px of horizontal movement, and
  // `failOffsetY` immediately yields to the parent vertical ScrollView for
  // any vertical motion. Without these, the chart "swallows" all touches and
  // either nothing happens or vertical scroll feels blocked.
  const pan = Gesture.Pan()
    .activeOffsetX([-8, 8])
    .failOffsetY([-15, 15])
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
      // Anchor zoom around the right edge — feels like price-action zoom.
      const rightEdge = savedStart.value + savedCount.value;
      const nextStart = Math.max(0, Math.min(total - nextCount, rightEdge - nextCount));
      runOnJS(commitView)(nextStart, nextCount);
    });

  const gesture = Gesture.Simultaneous(pinch, pan);

  // ─── Visible window ─────────────────────────────────────────────────────
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

  // ─── Geometry ───────────────────────────────────────────────────────────
  const geom = useMemo(() => {
    if (!visibleCandles || visibleCandles.length === 0) {
      return null;
    }
    const cs = visibleCandles;
    const lows = cs.map((c) => c.low);
    const highs = cs.map((c) => c.high);
    let lo = Math.min(...lows);
    let hi = Math.max(...highs);
    const padR = (hi - lo) * 0.05 || hi * 0.001 || 1;
    lo -= padR; hi += padR;
    const range = (hi - lo) || 1;

    const slotW = innerW / cs.length;
    const bodyW = Math.max(Math.min(slotW * 0.72, 12), 1.5);
    const xForIdx = (i: number) => padLeft + slotW * (i + 0.5);
    const yForPrice = (p: number) => padTop + candleH - ((p - lo) / range) * candleH;

    const timeMin = cs[0].time;
    const timeMax = cs[cs.length - 1].time;
    const timeSpan = (timeMax - timeMin) || 1;
    const xForTime = (t: number) => {
      const clamped = Math.max(timeMin, Math.min(timeMax, t));
      return padLeft + ((clamped - timeMin) / timeSpan) * innerW;
    };

    const maxVol = Math.max(1, ...cs.map((c) => c.volume || 0));

    const paths = cs.map((c, i) => {
      const x = xForIdx(i);
      const yHigh = yForPrice(c.high);
      const yLow = yForPrice(c.low);
      const yOpen = yForPrice(c.open);
      const yClose = yForPrice(c.close);
      const up = c.close >= c.open;
      const top = Math.min(yOpen, yClose);
      const bodyH = Math.max(Math.abs(yOpen - yClose), 1);
      const vRatio = (c.volume || 0) / maxVol;
      const vH = Math.max(1, vRatio * (volH - 4));
      return {
        x, yHigh, yLow, top, bodyH, bodyW, up,
        vTop: volTop + (volH - vH),
        vH,
        time: c.time,
      };
    });

    return { paths, yMin: lo, yMax: hi, range, xForTime, yForPrice, timeMin, timeMax, timeSpan };
  }, [visibleCandles, innerW, candleH, padLeft, padTop, volTop, volH]);

  if (total === 0 || !geom) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>No price data available.</Text>
      </View>
    );
  }

  const { paths, yMin, yMax, range, xForTime, yForPrice, timeMin, timeMax, timeSpan } = geom;
  const lastClose = visibleCandles[visibleCandles.length - 1].close;
  const firstOpen = visibleCandles[0].open;
  const change = lastClose - firstOpen;
  const changePct = (change / firstOpen) * 100;
  const liveY = livePrice != null ? yForPrice(livePrice) : null;
  const liveValid = liveY != null && Number.isFinite(liveY) && liveY >= padTop && liveY <= padTop + candleH;

  // Right-axis labels — 5 evenly spaced ticks
  const axisTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const price = yMax - (yMax - yMin) * f;
    const y = padTop + candleH * f;
    return { price, y };
  });

  // Time axis labels — 4 evenly spaced
  const formatTime = (t: number) => {
    const d = new Date(t * 1000);
    const tfLower = (tf || '').toLowerCase();
    const isIntraday = tfLower.endsWith('m') || tfLower === '1h' || tfLower === '4h';
    if (isIntraday) {
      const hh = d.getHours().toString().padStart(2, '0');
      const mm = d.getMinutes().toString().padStart(2, '0');
      return `${hh}:${mm}`;
    }
    return `${d.getDate()}/${d.getMonth() + 1}`;
  };
  const timeTicks = [0.05, 0.35, 0.65, 0.95].map((f) => ({
    x: padLeft + innerW * f,
    label: formatTime(timeMin + timeSpan * f),
  }));

  const fmtAxisPrice = (p: number) => {
    if (p >= 10000) return p.toLocaleString(undefined, { maximumFractionDigits: 0 });
    if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 1 });
    if (p >= 1) return p.toFixed(2);
    if (p >= 0.01) return p.toFixed(4);
    return p.toFixed(6);
  };

  const liveColor = change >= 0 ? colors.positive : colors.negative;

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
          {/* horizontal grid */}
          {axisTicks.map((t, i) => (
            <Line
              key={`g-${i}`}
              x1={padLeft} y1={t.y} x2={padLeft + innerW} y2={t.y}
              stroke={colors.border} strokeWidth={1} strokeDasharray="2,4" opacity={0.35}
            />
          ))}

          {/* FVG / liquidity zones — UNDER candles so wicks stay visible */}
          {zones.map((z, i) => {
            const x1 = xForTime(z.fromTime);
            const x2 = z.toTime ? xForTime(z.toTime) : (padLeft + innerW);
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
                  opacity={z.dim ? 0.10 : 0.20}
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

          {/* horizontal price lines (order-book walls) */}
          {priceLines.map((pl, i) => {
            const y = yForPrice(pl.price);
            if (!Number.isFinite(y) || y < padTop || y > padTop + candleH) return null;
            const stroke = pl.side === 'buy' ? colors.positive : colors.negative;
            return (
              <G key={`pl-${i}`}>
                <Line
                  x1={padLeft} y1={y} x2={padLeft + innerW} y2={y}
                  stroke={stroke} strokeWidth={1.5} opacity={0.7}
                  strokeDasharray="6,4"
                />
                {pl.label ? (
                  <G>
                    <Rect
                      x={padLeft + innerW + 2}
                      y={y - 7}
                      width={axisRightW - 4}
                      height={14}
                      fill="rgba(0,0,0,0.55)"
                      stroke={stroke}
                      strokeWidth={1}
                      rx={3}
                    />
                    <SvgText
                      x={padLeft + innerW + axisRightW / 2}
                      y={y + 3.5}
                      fontSize={9}
                      fontWeight="bold"
                      fill={stroke}
                      textAnchor="middle"
                    >
                      {pl.label}
                    </SvgText>
                  </G>
                ) : null}
              </G>
            );
          })}

          {/* candles */}
          {paths.map((p, i) => (
            <G key={`c-${i}`}>
              <Line
                x1={p.x} y1={p.yHigh} x2={p.x} y2={p.yLow}
                stroke={p.up ? colors.positive : colors.negative} strokeWidth={1}
              />
              <Rect
                x={p.x - p.bodyW / 2}
                y={p.top}
                width={p.bodyW}
                height={p.bodyH}
                fill={p.up ? colors.positive : colors.negative}
                rx={0.5}
              />
            </G>
          ))}

          {/* volume bars */}
          {showVolume ? (
            <G>
              <Line
                x1={padLeft} y1={volTop} x2={padLeft + innerW} y2={volTop}
                stroke={colors.border} strokeWidth={1} opacity={0.4}
              />
              {paths.map((p, i) => (
                <Rect
                  key={`v-${i}`}
                  x={p.x - p.bodyW / 2}
                  y={p.vTop}
                  width={p.bodyW}
                  height={p.vH}
                  fill={p.up ? colors.positive : colors.negative}
                  opacity={0.35}
                  rx={0.5}
                />
              ))}
            </G>
          ) : null}

          {/* right-axis price ticks */}
          {axisTicks.map((t, i) => (
            <SvgText
              key={`ax-${i}`}
              x={padLeft + innerW + 4}
              y={t.y + 3}
              fontSize={9}
              fill={colors.textMute}
              textAnchor="start"
            >
              {fmtAxisPrice(t.price)}
            </SvgText>
          ))}

          {/* time axis labels */}
          {timeTicks.map((t, i) => (
            <SvgText
              key={`tx-${i}`}
              x={t.x}
              y={timeTop + 11}
              fontSize={9}
              fill={colors.textMute}
              textAnchor="middle"
            >
              {t.label}
            </SvgText>
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

          {/* live-price horizontal line + right-edge tag (TradingView-style) */}
          {liveValid ? (
            <G>
              <Line
                x1={padLeft} y1={liveY!} x2={padLeft + innerW} y2={liveY!}
                stroke={liveColor} strokeWidth={1} opacity={0.6}
                strokeDasharray="3,3"
              />
              <Rect
                x={padLeft + innerW + 2}
                y={liveY! - 8}
                width={axisRightW - 4}
                height={16}
                fill={liveColor}
                rx={3}
              />
              <SvgText
                x={padLeft + innerW + axisRightW / 2}
                y={liveY! + 3.5}
                fontSize={10}
                fontWeight="bold"
                fill="#04111a"
                textAnchor="middle"
              >
                {fmtAxisPrice(livePrice!)}
              </SvgText>
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
    right: 60,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(255,255,255,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
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
});
