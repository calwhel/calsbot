import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, radius, spacing } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent';

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
  const valueColor =
    tone === 'positive' ? colors.positive :
    tone === 'negative' ? colors.negative :
    tone === 'accent'   ? colors.accent   :
    colors.text;

  return (
    <View style={[styles.card, compact && styles.compact]}>
      <Text style={styles.label} numberOfLines={1}>{label}</Text>
      <Text style={[styles.value, { color: valueColor }, compact && styles.valueCompact]} numberOfLines={1}>
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
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    minHeight: 92,
  },
  compact: {
    minHeight: 72,
    padding: spacing.md,
  },
  label: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: '600',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  value: {
    color: colors.text,
    fontSize: 22,
    fontWeight: '700',
  },
  valueCompact: {
    fontSize: 18,
  },
  sub: {
    color: colors.textMute,
    fontSize: 11,
    marginTop: 4,
  },
});
