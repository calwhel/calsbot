import React from 'react';
import { View, StyleSheet, ViewStyle, StyleProp } from 'react-native';
import { colors, radius, shadow } from '@/constants/colors';

/**
 * GradientCard (modern-dark) — flat surface card with hairline border.
 * Gradient/glow have been removed; the props are preserved as no-ops so
 * existing screens compile without churn.
 */
export function GradientCard({
  children,
  style,
  glowTone: _glowTone,
  highlight: _highlight = true,
}: {
  children: React.ReactNode;
  style?: StyleProp<ViewStyle>;
  glowTone?: 'accent' | 'positive' | 'none';
  highlight?: boolean;
}) {
  return (
    <View style={[styles.shadow, style]}>
      <View style={styles.wrap}>{children}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  shadow: {
    borderRadius: radius.xl,
    backgroundColor: colors.card,
    ...shadow.card,
  },
  wrap: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    overflow: 'hidden',
  },
});
