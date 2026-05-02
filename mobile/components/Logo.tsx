import React from 'react';
import { View, StyleSheet } from 'react-native';
import Svg, { Defs, LinearGradient, RadialGradient, Stop, Rect, Path, Circle, G } from 'react-native-svg';
import { glow, radius } from '@/constants/colors';

/**
 * TradeHub brand mark — a stylised candlestick + trend line in a rounded
 * tile with a cyan→indigo gradient and a soft accent glow.
 *
 * The mark is fully self-contained SVG so it stays sharp at every size and
 * doesn't depend on a raster icon file.
 */
export function Logo({ size = 56, withGlow = true }: { size?: number; withGlow?: boolean }) {
  // Unique IDs per instance so multiple Logos on the same screen don't collide
  // with each other's <Defs> (RN-Web + some native paths resolve url(#…)
  // globally, not per-Svg).
  const uid = React.useId().replace(/:/g, '');
  const tileId = `tile-${uid}`;
  const markId = `mark-${uid}`;
  const apexId = `apex-${uid}`;

  return (
    <View
      style={[
        { width: size, height: size, borderRadius: size * 0.28 },
        withGlow && glow.accent,
        styles.wrap,
      ]}
    >
      <Svg width={size} height={size} viewBox="0 0 64 64">
        <Defs>
          <LinearGradient id={tileId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#1a2240" />
            <Stop offset="1" stopColor="#0a0f20" />
          </LinearGradient>
          <LinearGradient id={markId} x1="0" y1="1" x2="1" y2="0">
            <Stop offset="0" stopColor="#22d3ee" />
            <Stop offset="0.55" stopColor="#3b82f6" />
            <Stop offset="1" stopColor="#a78bfa" />
          </LinearGradient>
          <RadialGradient id={apexId} cx="0.5" cy="0.5" r="0.5">
            <Stop offset="0" stopColor="#67e8f9" stopOpacity="0.9" />
            <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </RadialGradient>
        </Defs>

        {/* Tile */}
        <Rect x="0" y="0" width="64" height="64" rx="18" fill={`url(#${tileId})`} />
        {/* Subtle inner highlight border */}
        <Rect
          x="0.75" y="0.75" width="62.5" height="62.5" rx="17.25"
          fill="none" stroke="rgba(34,211,238,0.22)" strokeWidth="1.2"
        />

        {/* Faint price-action candles in the background */}
        <G opacity="0.32">
          <Rect x="11" y="38" width="3.5" height="14" rx="1.4" fill="#22d3ee" />
          <Rect x="20" y="30" width="3.5" height="22" rx="1.4" fill="#22d3ee" />
          <Rect x="29" y="34" width="3.5" height="18" rx="1.4" fill="#22d3ee" />
          <Rect x="38" y="24" width="3.5" height="28" rx="1.4" fill="#22d3ee" />
        </G>

        {/* Trend line */}
        <Path
          d="M 12 46 L 24 34 L 32 40 L 47 18"
          stroke={`url(#${markId})`}
          strokeWidth="4"
          strokeLinecap="round"
          strokeLinejoin="round"
          fill="none"
        />

        {/* Apex glow + dot */}
        <Circle cx="47" cy="18" r="9" fill={`url(#${apexId})`} />
        <Circle cx="47" cy="18" r="3.6" fill="#e0f7ff" />
        <Circle cx="47" cy="18" r="2.2" fill="#22d3ee" />
      </Svg>
    </View>
  );
}

/** Wordmark — logo + "TradeHub" text. Used on login + settings header. */
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

// Re-export for convenience
export { radius };
