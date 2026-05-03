import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font, radius, shadow, spacing } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent' | 'warning';

const TONE_COLOR: Record<Tone, string> = {
  neutral:  colors.text,
  positive: colors.positive,
  negative: colors.negative,
  accent:   colors.text, // 'accent' is action/neutral — never implies +P&L
  warning:  colors.warning,
};

/**
 * StatCard (modern-dark) — flat metric card with hairline border. The
 * tone-coloured top stripe has been removed; tone now only affects the
 * value text colour.
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
  const valueColor = TONE_COLOR[tone];
  return (
    <View style={[styles.card, compact && styles.compact]}>
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
    ...shadow.card,
  },
  compact: {
    minHeight: 76,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 11,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    marginBottom: 8,
  },
  value: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 26,
    letterSpacing: -0.7,
    fontVariant: ['tabular-nums'],
  },
  valueCompact: {
    fontSize: 20,
    letterSpacing: -0.5,
  },
  sub: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    marginTop: 4,
  },
});
