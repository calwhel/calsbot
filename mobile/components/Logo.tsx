import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Path, Rect, G } from 'react-native-svg';
import { colors, radius } from '@/constants/colors';

/**
 * TradeHub brand mark — modern-dark monochrome version.
 *
 * A stylised candlestick + trend line in a flat graphite tile. No gradients,
 * no glow. The mark reads as a tool, not as a logo demo.
 */
export function Logo({ size = 56, withGlow: _withGlow = true }: { size?: number; withGlow?: boolean }) {
  return (
    <View
      style={[
        { width: size, height: size, borderRadius: size * 0.24 },
        styles.wrap,
      ]}
    >
      <Svg width={size} height={size} viewBox="0 0 64 64">
        {/* Tile */}
        <Rect x="0" y="0" width="64" height="64" rx="14" fill={colors.cardHi} />
        <Rect
          x="0.5" y="0.5" width="63" height="63" rx="13.5"
          fill="none" stroke={colors.borderHi} strokeWidth="1"
        />

        {/* Faint price-action candles */}
        <G opacity="0.35">
          <Rect x="11" y="38" width="3.5" height="14" rx="1" fill={colors.textDim} />
          <Rect x="20" y="30" width="3.5" height="22" rx="1" fill={colors.textDim} />
          <Rect x="29" y="34" width="3.5" height="18" rx="1" fill={colors.textDim} />
          <Rect x="38" y="24" width="3.5" height="28" rx="1" fill={colors.textDim} />
        </G>

        {/* Trend line — single restrained green */}
        <Path
          d="M 12 46 L 24 34 L 32 40 L 47 18"
          stroke={colors.positive}
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />
      </Svg>
    </View>
  );
}

/** Wordmark — logo only (text rendered separately when needed). */
export function Wordmark({ size = 40 }: { size?: number }) {
  return (
    <View style={styles.wordRow}>
      <Logo size={size} />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    overflow: 'hidden',
    backgroundColor: 'transparent',
  },
  wordRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
});

export { radius };
