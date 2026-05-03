import React from 'react';
import { Pressable, StyleSheet, Text, View, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';

type Tone = 'accent' | 'positive' | 'violet' | 'gold' | 'magenta' | 'mint';

/**
 * ActionTile (modern-dark) — flat icon button used in the Home quick-actions
 * row. The per-tone gradient circles have been removed; every tile renders
 * as a neutral chip with a high-contrast text label below. Tone is preserved
 * in the API but no longer drives colour.
 */
export function ActionTile({
  icon,
  label,
  tone: _tone = 'accent',
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  tone?: Tone;
  onPress: () => void;
}) {
  const handle = () => {
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    onPress();
  };

  return (
    <Pressable
      onPress={handle}
      style={({ pressed }) => [
        styles.tile,
        pressed && { opacity: 0.7 },
      ]}
    >
      <View style={styles.iconWrap}>
        <Ionicons name={icon} size={20} color={colors.text} />
      </View>
      <Text style={styles.label} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  tile: {
    width: 78,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  iconWrap: {
    width: 44,
    height: 44,
    borderRadius: radius.md,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 11,
    letterSpacing: 0.1,
    textAlign: 'center',
  },
});
