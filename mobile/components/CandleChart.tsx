import React, { useMemo } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Line, Rect, Circle, G } from 'react-native-svg';
import { colors, font, radius, spacing } from '@/constants/colors';

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
}) {
  const padX = 8;
  const padY = 14;
  const innerW = Math.max(width - padX * 2, 1);
  const innerH = Math.max(height - padY * 2, 1);

  const { paths, yMin, yMax, xForTime, yForPrice } = useMemo(() => {
    if (!candles || candles.length === 0) {
      return { paths: [] as any[], yMin: 0, yMax: 1, xForTime: () => 0, yForPrice: () => 0 };
    }
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

  if (!candles || candles.length === 0) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>No price data available.</Text>
      </View>
    );
  }

  const lastClose = candles[candles.length - 1].close;
  const firstOpen = candles[0].open;
  const change = lastClose - firstOpen;
  const changePct = (change / firstOpen) * 100;

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
                opacity={z.dim ? 0.06 : 0.14}
              />
              <Line
                x1={x} y1={yTop} x2={x + w} y2={yTop}
                stroke={baseFill} strokeWidth={0.75} opacity={z.dim ? 0.25 : 0.55}
                strokeDasharray="2,3"
              />
              <Line
                x1={x} y1={yBot} x2={x + w} y2={yBot}
                stroke={baseFill} strokeWidth={0.75} opacity={z.dim ? 0.25 : 0.55}
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
              stroke={stroke} strokeWidth={1} opacity={0.45}
              strokeDasharray="4,4"
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
      </Svg>
      {showOhlcLegend ? (
        <View style={styles.ohlcOverlay} pointerEvents="none">
          {(() => {
            const c = candles[candles.length - 1];
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
