import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, spacing } from '@/constants/colors';

/**
 * SectionLabel (modern-dark) — quiet uppercase header. The leading colour
 * bar has been removed. Right-side action stays, but always renders in the
 * neutral text-dim tone (no per-tone tinting).
 */
export function SectionLabel({
  label,
  caption,
  tone: _tone = 'accent',
  actionLabel,
  onActionPress,
}: {
  label: string;
  caption?: string;
  tone?: 'accent' | 'positive' | 'violet' | 'warning' | 'neutral';
  actionLabel?: string;
  onActionPress?: () => void;
}) {
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        <Text style={styles.label} numberOfLines={1}>{label}</Text>
        <View style={{ flex: 1 }} />
        {actionLabel && onActionPress ? (
          <Pressable
            onPress={onActionPress}
            hitSlop={8}
            style={({ pressed }) => [
              styles.action,
              pressed && { opacity: 0.6 },
            ]}
          >
            <Text style={styles.actionText}>{actionLabel}</Text>
            <Ionicons name="chevron-forward" size={13} color={colors.textDim} />
          </Pressable>
        ) : null}
      </View>
      {caption ? <Text style={styles.caption}>{caption}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    marginBottom: spacing.md,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  label: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  action: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  actionText: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 12,
    letterSpacing: 0.1,
  },
  caption: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    marginTop: 4,
  },
});
