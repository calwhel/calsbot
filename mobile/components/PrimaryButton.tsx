import React from 'react';
import { Pressable, Text, StyleSheet, ActivityIndicator, View, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient, Stop, Rect } from 'react-native-svg';
import { colors, font, glow, radius } from '@/constants/colors';

type Variant = 'primary' | 'secondary' | 'destructive' | 'ghost';

export function PrimaryButton({
  label,
  onPress,
  loading = false,
  disabled = false,
  variant = 'primary',
  icon,
}: {
  label: string;
  onPress: () => void;
  loading?: boolean;
  disabled?: boolean;
  variant?: Variant;
  /** Optional leading element (e.g. an Ionicon). */
  icon?: React.ReactNode;
}) {
  const isDisabled = disabled || loading;
  // Per-instance ID so multiple primary buttons on a screen don't share defs.
  const uid = React.useId().replace(/:/g, '');
  const gradId = `pb-grad-${uid}`;

  const handlePress = () => {
    if (isDisabled) return;
    if (Platform.OS !== 'web') {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    onPress();
  };

  // Primary uses a gradient + glow; others use solid surfaces.
  const isPrimary = variant === 'primary';
  const palette =
    variant === 'destructive'
      ? { fg: colors.negative, bg: colors.negativeDim, border: 'transparent' }
      : variant === 'secondary'
        ? { fg: colors.text, bg: colors.cardHi, border: colors.borderHi }
        : variant === 'ghost'
          ? { fg: colors.textDim, bg: 'transparent', border: colors.border }
          : { fg: colors.accentText, bg: colors.accent, border: 'transparent' };

  return (
    <Pressable
      onPress={handlePress}
      disabled={isDisabled}
      style={({ pressed }) => [
        styles.btn,
        { backgroundColor: palette.bg, borderColor: palette.border, borderWidth: variant === 'ghost' || variant === 'secondary' ? 1 : 0 },
        isPrimary && glow.accent,
        pressed && !isDisabled && { transform: [{ scale: 0.985 }], opacity: 0.92 },
        isDisabled && { opacity: 0.5 },
      ]}
    >
      {isPrimary ? (
        <Svg style={StyleSheet.absoluteFill}>
          <Defs>
            <LinearGradient id={gradId} x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0" stopColor="#22d3ee" />
              <Stop offset="0.55" stopColor="#0ea5e9" />
              <Stop offset="1" stopColor="#6366f1" />
            </LinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill={`url(#${gradId})`} />
        </Svg>
      ) : null}
      <View style={styles.row}>
        {loading ? (
          <ActivityIndicator color={palette.fg} />
        ) : (
          <>
            {icon ? <View style={{ marginRight: 6 }}>{icon}</View> : null}
            <Text style={[styles.text, { color: palette.fg }]}>{label}</Text>
          </>
        )}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  btn: {
    height: 52,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 18,
  },
  text: {
    fontFamily: font.bold,
    fontSize: 15,
    letterSpacing: 0.2,
  },
});
