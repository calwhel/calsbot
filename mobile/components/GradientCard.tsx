import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import Svg, { Defs, LinearGradient, Stop, Rect } from 'react-native-svg';
import { colors, glow, radius } from '@/constants/colors';

/**
 * A card surface with a built-in vertical gradient + 1px highlight on top
 * edge, and an optional accent-coloured glow shadow. Children render above
 * the gradient.
 *
 * Layout: width comes from style, height from contents. The svg fills via
 * StyleSheet.absoluteFill so it follows the rendered size automatically.
 */
export function GradientCard({
  children,
  style,
  glowTone,
  highlight = true,
}: {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  /** Add a coloured glow shadow underneath the card. */
  glowTone?: 'accent' | 'positive' | 'none';
  /** Show the 1px gradient highlight along the top edge. */
  highlight?: boolean;
}) {
  const glowStyle =
    glowTone === 'accent' ? glow.accent :
    glowTone === 'positive' ? glow.positive :
    null;

  // Per-instance ID so multiple GradientCards on a screen don't collide.
  const uid = React.useId().replace(/:/g, '');
  const bgId = `gc-bg-${uid}`;

  return (
    <View style={[styles.wrap, glowStyle, style]}>
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <LinearGradient id={bgId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor="#1a2238" />
            <Stop offset="0.55" stopColor="#141b2e" />
            <Stop offset="1" stopColor="#0f1524" />
          </LinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill={`url(#${bgId})`} />
      </Svg>
      {highlight ? <View style={styles.highlight} /> : null}
      <View style={{ position: 'relative' }}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    backgroundColor: colors.card, // fallback while svg paints
  },
  highlight: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.06)',
  },
});
