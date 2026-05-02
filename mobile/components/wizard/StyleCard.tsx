import React from 'react';
import { Pressable, View, Text, StyleSheet, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * Tappable preset card used on Step 1 of the wizard.
 * Three across on phone, with an icon, label, and one-line tagline.
 * Selected state shows the cyan accent border + soft glow.
 */
export function StyleCard({
  icon,
  label,
  tagline,
  selected,
  onPress,
}: {
  icon: string;
  label: string;
  tagline: string;
  selected: boolean;
  onPress: () => void;
}) {
  const handlePress = () => {
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    onPress();
  };
  return (
    <Pressable
      onPress={handlePress}
      style={({ pressed }) => [
        styles.card,
        selected && styles.cardActive,
        pressed && { opacity: 0.85 },
      ]}
      accessibilityRole="button"
      accessibilityState={{ selected }}
      accessibilityLabel={`${label} trading style`}
    >
      <Text style={styles.icon}>{icon}</Text>
      <Text style={[styles.label, selected && styles.labelActive]} numberOfLines={1}>
        {label}
      </Text>
      <Text style={styles.tagline} numberOfLines={2}>
        {tagline}
      </Text>
      {selected ? <View style={styles.checkDot} /> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  card: {
    flexBasis: '31%',
    flexGrow: 1,
    minHeight: 110,
    borderRadius: radius.lg,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.sm,
    alignItems: 'center',
    justifyContent: 'flex-start',
    position: 'relative',
  },
  cardActive: {
    borderColor: colors.accent,
    backgroundColor: colors.cardHi,
  },
  icon: {
    fontSize: 22,
    marginBottom: 4,
  },
  label: {
    fontFamily: font.semibold,
    fontSize: 13,
    color: colors.text,
    textAlign: 'center',
    marginBottom: 2,
  },
  labelActive: { color: colors.accent },
  tagline: {
    fontFamily: font.regular,
    fontSize: 10.5,
    color: colors.textMute,
    textAlign: 'center',
    lineHeight: 14,
  },
  checkDot: {
    position: 'absolute',
    top: 6, right: 6,
    width: 8, height: 8, borderRadius: 4,
    backgroundColor: colors.accent,
  },
});
