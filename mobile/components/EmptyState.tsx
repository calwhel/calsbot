import React from 'react';
import { View, Text, StyleSheet, Pressable, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Defs, RadialGradient, Stop, Circle } from 'react-native-svg';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';

export function EmptyState({
  icon = 'cube-outline',
  title,
  hint,
  tone = 'neutral',
  ctaLabel,
  onCta,
}: {
  icon?: keyof typeof Ionicons.glyphMap;
  title: string;
  hint?: string;
  tone?: 'neutral' | 'accent';
  ctaLabel?: string;
  onCta?: () => void;
}) {
  const orbColor = tone === 'accent' ? colors.accent : colors.violet;
  const iconColor = tone === 'accent' ? colors.accent : colors.textDim;
  const handleCta = () => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    onCta?.();
  };
  // Per-instance ID so two EmptyStates in the same render tree don't collide.
  const uid = React.useId().replace(/:/g, '');
  const orbId = `orb-${uid}`;

  return (
    <View style={styles.wrap}>
      <View style={styles.orb}>
        <Svg width={88} height={88} style={StyleSheet.absoluteFill}>
          <Defs>
            <RadialGradient id={orbId} cx="0.5" cy="0.5" r="0.5">
              <Stop offset="0" stopColor={orbColor} stopOpacity="0.25" />
              <Stop offset="1" stopColor={orbColor} stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="44" cy="44" r="42" fill={`url(#${orbId})`} />
        </Svg>
        <View style={styles.orbInner}>
          <Ionicons name={icon} size={36} color={iconColor} />
        </View>
      </View>
      <Text style={styles.title}>{title}</Text>
      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
      {ctaLabel && onCta ? (
        <Pressable
          onPress={handleCta}
          style={({ pressed }) => [styles.cta, pressed && { opacity: 0.85 }]}
        >
          <Text style={styles.ctaText}>{ctaLabel}</Text>
        </Pressable>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xxl,
    paddingHorizontal: spacing.xl,
  },
  orb: {
    width: 88,
    height: 88,
    borderRadius: 44,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  orbInner: {
    width: 64,
    height: 64,
    borderRadius: radius.lg,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.borderHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 17,
    marginTop: spacing.sm,
    textAlign: 'center',
    letterSpacing: -0.2,
  },
  hint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 13,
    marginTop: 6,
    textAlign: 'center',
    lineHeight: 19,
    maxWidth: 280,
  },
  cta: {
    marginTop: spacing.lg,
    backgroundColor: colors.text,
    paddingHorizontal: 20,
    height: 44,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
  },
  ctaText: {
    color: '#0E0F11',
    fontFamily: font.semibold,
    fontSize: 14,
    letterSpacing: 0.1,
  },
});
