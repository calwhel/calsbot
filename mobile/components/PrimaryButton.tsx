import React from 'react';
import { Pressable, Text, StyleSheet, ActivityIndicator, View, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, shadow } from '@/constants/colors';

type Variant = 'primary' | 'secondary' | 'destructive' | 'ghost';

/**
 * PrimaryButton (modern-dark) — solid surfaces, no glow, no SVG gradient.
 *   primary     → high-contrast white surface, dark label (Cash App style)
 *   secondary   → elevated grey surface, white label
 *   destructive → red-tinted surface, red label
 *   ghost       → transparent with hairline border
 */
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
  icon?: React.ReactNode;
}) {
  const isDisabled = disabled || loading;

  const handlePress = () => {
    if (isDisabled) return;
    if (Platform.OS !== 'web') {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    }
    onPress();
  };

  const palette =
    variant === 'destructive'
      ? { fg: colors.negative, bg: colors.negativeDim, border: 'transparent' }
      : variant === 'secondary'
        ? { fg: colors.text, bg: colors.cardHi, border: colors.border }
        : variant === 'ghost'
          ? { fg: colors.text, bg: 'transparent', border: colors.borderHi }
          : { fg: '#0E0F11', bg: colors.text, border: 'transparent' };

  return (
    <Pressable
      onPress={handlePress}
      disabled={isDisabled}
      style={({ pressed }) => [
        styles.btn,
        {
          backgroundColor: palette.bg,
          borderColor: palette.border,
          borderWidth: variant === 'ghost' || variant === 'secondary' ? 1 : 0,
        },
        variant === 'primary' && shadow.lift,
        pressed && !isDisabled && { opacity: 0.85 },
        isDisabled && { opacity: 0.5 },
      ]}
    >
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
    height: 56,
    borderRadius: radius.lg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
  },
  text: {
    fontFamily: font.semibold,
    fontSize: 16,
    letterSpacing: 0.1,
  },
});
