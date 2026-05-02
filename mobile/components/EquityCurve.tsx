import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Path, Line, Defs, LinearGradient, Stop } from 'react-native-svg';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * SVG line chart for cumulative P&L (equity curve).
 * Auto-scales Y to data range, draws zero baseline, fills under the curve.
 */
export function EquityCurve({
  values,
  width,
  height = 160,
  title,
}: {
  values: number[];
  width: number;
  height?: number;
  title?: string;
}) {
  // Hooks must run unconditionally — declare the per-instance ID BEFORE the
  // early-return guard so React's hook order stays stable between renders
  // (values can flip between <2 and ≥2 as data loads).
  const uid = React.useId().replace(/:/g, '');
  const fillId = `eqfill-${uid}`;

  const padX = 8;
  const padY = 14;
  const innerW = Math.max(width - padX * 2, 1);
  const innerH = Math.max(height - padY * 2, 1);

  if (!values || values.length < 2) {
    return (
      <View style={[styles.empty, { height }]}>
        <Text style={styles.emptyText}>
          {title || 'Not enough trades for an equity curve yet.'}
        </Text>
      </View>
    );
  }

  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const range = (max - min) || 1;

  const xStep = innerW / (values.length - 1);
  const points = values.map((v, i) => {
    const x = padX + i * xStep;
    const y = padY + innerH - ((v - min) / range) * innerH;
    return { x, y };
  });

  const linePath = points
    .map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`))
    .join(' ');

  const fillPath =
    `${linePath} L ${points[points.length - 1].x} ${padY + innerH} L ${points[0].x} ${padY + innerH} Z`;

  const last = values[values.length - 1];
  const stroke = last >= 0 ? colors.positive : colors.negative;

  // zero baseline (only if 0 falls inside the range)
  let zeroY: number | null = null;
  if (min < 0 && max > 0) {
    zeroY = padY + innerH - ((0 - min) / range) * innerH;
  }

  return (
    <View style={styles.wrap}>
      <Svg width={width} height={height}>
        <Defs>
          <LinearGradient id={fillId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={stroke} stopOpacity={0.35} />
            <Stop offset="1" stopColor={stroke} stopOpacity={0.0} />
          </LinearGradient>
        </Defs>
        {zeroY !== null && (
          <Line
            x1={padX} y1={zeroY} x2={width - padX} y2={zeroY}
            stroke={colors.border} strokeWidth={1} strokeDasharray="4,4"
          />
        )}
        <Path d={fillPath} fill={`url(#${fillId})`} />
        <Path d={linePath} fill="none" stroke={stroke} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
      </Svg>
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
