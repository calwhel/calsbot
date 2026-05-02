import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font, radius } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent' | 'warning' | 'violet';

export function Pill({
  label,
  tone = 'neutral',
  small = false,
}: {
  label: string;
  tone?: Tone;
  small?: boolean;
}) {
  const palette = {
    neutral:  { bg: colors.pillBg,     fg: colors.textDim,  border: colors.border },
    positive: { bg: colors.positiveDim, fg: colors.positive, border: 'rgba(52,211,153,0.32)' },
    negative: { bg: colors.negativeDim, fg: colors.negative, border: 'rgba(248,113,113,0.32)' },
    accent:   { bg: colors.accentDim,   fg: colors.accent,   border: 'rgba(34,211,238,0.36)' },
    warning:  { bg: colors.warningDim,  fg: colors.warning,  border: 'rgba(251,191,36,0.32)' },
    violet:   { bg: colors.violetDim,   fg: colors.violet,   border: 'rgba(167,139,250,0.32)' },
  }[tone];

  return (
    <View style={[
      styles.pill,
      small && styles.pillSmall,
      { backgroundColor: palette.bg, borderColor: palette.border },
    ]}>
      <Text style={[styles.text, small && styles.textSmall, { color: palette.fg }]}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  pill: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.pill,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  pillSmall: {
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  text: {
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  textSmall: {
    fontSize: 10,
  },
});
