import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Defs, LinearGradient, Stop, Rect } from 'react-native-svg';
import { colors, font, radius, spacing } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent' | 'warning';

const TONE_COLOR: Record<Tone, string> = {
  neutral:  colors.text,
  positive: colors.positive,
  negative: colors.negative,
  accent:   colors.accent,
  warning:  colors.warning,
};

const TONE_STRIPE: Record<Tone, [string, string] | null> = {
  neutral:  null,
  positive: [colors.positive, 'rgba(52,211,153,0)'],
  negative: [colors.negative, 'rgba(248,113,113,0)'],
  accent:   [colors.accent,   'rgba(34,211,238,0)'],
  warning:  [colors.warning,  'rgba(251,191,36,0)'],
};

/**
 * Stat card with a tone-coloured top stripe (gradient that fades horizontally).
 * Used in dense grids on Home and Strategy detail screens.
 */
export function StatCard({
  label,
  value,
  sub,
  tone = 'neutral',
  compact = false,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: Tone;
  compact?: boolean;
}) {
  const stripe = TONE_STRIPE[tone];
  const valueColor = TONE_COLOR[tone];
  // Per-instance ID so multiple StatCards (e.g. a row of 4) with the same tone
  // don't collide on a single SVG def.
  const uid = React.useId().replace(/:/g, '');
  const stripeId = `stat-${tone}-${uid}`;

  return (
    <View style={[styles.card, compact && styles.compact]}>
      {stripe ? (
        <Svg style={styles.stripe} pointerEvents="none">
          <Defs>
            <LinearGradient id={stripeId} x1="0" y1="0" x2="1" y2="0">
              <Stop offset="0" stopColor={stripe[0]} stopOpacity={0.85} />
              <Stop offset="1" stopColor={stripe[1]} stopOpacity={0} />
            </LinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill={`url(#${stripeId})`} />
        </Svg>
      ) : null}
      <Text style={styles.label} numberOfLines={1}>{label}</Text>
      <Text
        style={[styles.value, { color: valueColor }, compact && styles.valueCompact]}
        numberOfLines={1}
      >
        {value}
      </Text>
      {sub ? <Text style={styles.sub} numberOfLines={1}>{sub}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    paddingVertical: spacing.lg,
    paddingHorizontal: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    minHeight: 96,
    overflow: 'hidden',
  },
  compact: {
    minHeight: 76,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  stripe: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 2.5,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  value: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 22,
    letterSpacing: -0.4,
    fontVariant: ['tabular-nums'],
  },
  valueCompact: {
    fontSize: 18,
  },
  sub: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
    marginTop: 4,
  },
});
