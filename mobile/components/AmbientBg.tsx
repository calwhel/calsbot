import React from 'react';
import { StyleSheet, View } from 'react-native';
import Svg, { Defs, RadialGradient, Stop, Rect } from 'react-native-svg';

/**
 * Subtle radial gradient blobs in the upper portion of the screen — a quiet
 * ambient backdrop that reads as "designed" without competing with content.
 * Renders absolutely-positioned and pointer-events: none so it never
 * interferes with touch handling.
 */
export function AmbientBg({
  variant = 'duo',
}: {
  variant?: 'duo' | 'cyan' | 'violet' | 'none';
}) {
  // Unique IDs per instance so multiple ambient backdrops in a session
  // (e.g. on tab switch with screen unmount delay) don't share defs.
  const uid = React.useId().replace(/:/g, '');
  const cyanId = `amb-cyan-${uid}`;
  const violetId = `amb-violet-${uid}`;

  if (variant === 'none') return null;

  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="xMidYMid slice">
        <Defs>
          <RadialGradient id={cyanId} cx="20%" cy="0%" rx="65%" ry="50%">
            <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.22" />
            <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </RadialGradient>
          <RadialGradient id={violetId} cx="100%" cy="18%" rx="60%" ry="55%">
            <Stop offset="0" stopColor="#8b5cf6" stopOpacity="0.15" />
            <Stop offset="1" stopColor="#8b5cf6" stopOpacity="0" />
          </RadialGradient>
        </Defs>
        {(variant === 'duo' || variant === 'cyan') && (
          <Rect width="100%" height="100%" fill={`url(#${cyanId})`} />
        )}
        {(variant === 'duo' || variant === 'violet') && (
          <Rect width="100%" height="100%" fill={`url(#${violetId})`} />
        )}
      </Svg>
    </View>
  );
}
