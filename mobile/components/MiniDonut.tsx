import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Circle, Defs, LinearGradient as SvgLinearGradient, Stop } from 'react-native-svg';
import { colors, font } from '@/constants/colors';

/**
 * MiniDonut — circular progress ring with a centred label.
 *
 *   ╭──────╮
 *   │  68% │
 *   ╰──────╯
 *
 * Used for win-rate visualisation on strategy cards. Stroke colour is
 * picked from `tone` (positive/warning/negative) — defaults to a tone
 * derived from the value itself (>=55 positive, <40 negative, else warning).
 */
export function MiniDonut({
  value,
  size = 44,
  stroke = 4.5,
  label,
  tone,
}: {
  /** 0..100 percentage to display */
  value: number;
  size?: number;
  stroke?: number;
  /** Override the centred label (defaults to `${value}%`). */
  label?: string;
  tone?: 'positive' | 'warning' | 'negative' | 'accent' | 'neutral';
}) {
  const uid = React.useId().replace(/:/g, '');
  const gradId = `donut-${uid}`;

  const pct = Math.max(0, Math.min(100, value));
  const auto: 'positive' | 'warning' | 'negative' =
    pct >= 55 ? 'positive' : pct < 40 ? 'negative' : 'warning';
  const finalTone = tone || auto;

  const colorMap: Record<typeof finalTone | 'accent' | 'neutral', readonly [string, string]> = {
    positive: [colors.positive, colors.positive] as const,
    warning:  [colors.warning, colors.warning] as const,
    negative: [colors.negative, colors.negative] as const,
    accent:   [colors.positive, colors.positive] as const,
    neutral:  [colors.textDim, colors.textDim] as const,
  };
  const [c0, c1] = colorMap[finalTone];

  const r = (size - stroke) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const dashOffset = circumference - (pct / 100) * circumference;

  return (
    <View style={[styles.wrap, { width: size, height: size }]}>
      <Svg width={size} height={size}>
        <Defs>
          <SvgLinearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor={c0} />
            <Stop offset="1" stopColor={c1} />
          </SvgLinearGradient>
        </Defs>
        {/* track */}
        <Circle
          cx={cx} cy={cy} r={r}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth={stroke}
          fill="none"
        />
        {/* progress */}
        <Circle
          cx={cx} cy={cy} r={r}
          stroke={`url(#${gradId})`}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          transform={`rotate(-90 ${cx} ${cy})`}
        />
      </Svg>
      <View style={styles.center} pointerEvents="none">
        <Text style={[styles.label, size <= 38 && styles.labelSm]}>
          {label ?? `${Math.round(pct)}%`}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { position: 'relative', alignItems: 'center', justifyContent: 'center' },
  center: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 11,
    letterSpacing: -0.2,
    fontVariant: ['tabular-nums'],
  },
  labelSm: { fontSize: 9.5 },
});
