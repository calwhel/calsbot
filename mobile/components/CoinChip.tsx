import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font, radius } from '@/constants/colors';

/**
 * CoinChip (modern-dark) — monochrome circular badge with the ticker's
 * first letter. The previous per-coin gradient identities have been
 * removed; every coin renders in the same calm graphite chip so the
 * lists read as data, not as a colour grid.
 */
export function CoinChip({
  symbol,
  size = 32,
}: {
  symbol: string;
  size?: number;
}) {
  const letter = (symbol.match(/[A-Za-z]/)?.[0] || '?').toUpperCase();
  return (
    <View
      style={[
        styles.wrap,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
        },
      ]}
    >
      <Text style={[styles.letter, { fontSize: size * 0.42 }]}>{letter}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.pill,
  },
  letter: {
    color: colors.text,
    fontFamily: font.semibold,
    letterSpacing: -0.2,
  },
});
