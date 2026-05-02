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

/**
 * Compact native candlestick chart. Renders the most recent N candles with
 * optional entry/exit markers overlayed at their (time, price) coordinates.
 *
 * Designed for the strategy detail screen — fixed height, fills width.
 */
export function CandleChart({
  candles,
  markers = [],
  width,
  height = 180,
  symbol,
  tf,
}: {
  candles: Candle[];
  markers?: ChartMarker[];
  width: number;
  height?: number;
  symbol?: string;
  tf?: string;
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
});
