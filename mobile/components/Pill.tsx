import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, radius } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent' | 'warning';

export function Pill({ label, tone = 'neutral', small = false }: { label: string; tone?: Tone; small?: boolean }) {
  const palette = {
    neutral:  { bg: colors.pillBg,     fg: colors.textDim },
    positive: { bg: colors.positiveDim, fg: colors.positive },
    negative: { bg: colors.negativeDim, fg: colors.negative },
    accent:   { bg: colors.accentDim,   fg: colors.accent },
    warning:  { bg: colors.warningDim,  fg: colors.warning },
  }[tone];

  return (
    <View style={[styles.pill, small && styles.pillSmall, { backgroundColor: palette.bg }]}>
      <Text style={[styles.text, small && styles.textSmall, { color: palette.fg }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.sm,
    alignSelf: 'flex-start',
  },
  pillSmall: {
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  text: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.4,
    textTransform: 'uppercase',
  },
  textSmall: {
    fontSize: 10,
  },
});
