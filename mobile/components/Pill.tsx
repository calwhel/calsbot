import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font, radius } from '@/constants/colors';

type Tone = 'neutral' | 'positive' | 'negative' | 'accent' | 'warning' | 'violet';

/**
 * Pill (modern-dark) — quiet status chip. Borderless, low-contrast surfaces.
 * Tones collapse to: positive (green), negative (red), warning (amber), and
 * neutral. Accent + violet both render as neutral.
 */
export function Pill({
  label,
  tone = 'neutral',
  small = false,
}: {
  label: string;
  tone?: Tone;
  small?: boolean;
}) {
  const palette = (() => {
    switch (tone) {
      case 'positive': return { bg: colors.positiveDim, fg: colors.positive };
      case 'negative': return { bg: colors.negativeDim, fg: colors.negative };
      case 'warning':  return { bg: colors.warningDim,  fg: colors.warning };
      case 'accent':   return { bg: colors.cardHi,      fg: colors.text };
      case 'violet':   return { bg: colors.cardHi,      fg: colors.textDim };
      default:         return { bg: colors.cardHi,      fg: colors.textDim };
    }
  })();

  return (
    <View style={[
      styles.pill,
      small && styles.pillSmall,
      { backgroundColor: palette.bg },
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
    alignSelf: 'flex-start',
  },
  pillSmall: {
    paddingHorizontal: 8,
    paddingVertical: 2,
  },
  text: {
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.3,
    textTransform: 'uppercase',
  },
  textSmall: {
    fontSize: 10,
  },
});
