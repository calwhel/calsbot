import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, spacing } from '@/constants/colors';

/**
 * SectionLabel — uppercase header used between content blocks.
 *
 *   ┃ RECENT TRADES                                        See all  ›
 *   ─────────────────────────────────────────────────────────────────
 *
 * The leading bar is a coloured gradient sliver that reinforces the
 * section's tone (defaults to brand cyan). Optional right-side action
 * link, optional caption underneath.
 */
export function SectionLabel({
  label,
  caption,
  tone = 'accent',
  actionLabel,
  onActionPress,
}: {
  label: string;
  caption?: string;
  tone?: 'accent' | 'positive' | 'violet' | 'warning' | 'neutral';
  actionLabel?: string;
  onActionPress?: () => void;
}) {
  const accentMap = {
    accent:   colors.accent,
    positive: colors.positive,
    violet:   colors.violet,
    warning:  colors.warning,
    neutral:  colors.textDim,
  };
  const accent = accentMap[tone];

  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        <View style={[styles.bar, { backgroundColor: accent }]} />
        <Text style={styles.label} numberOfLines={1}>{label}</Text>
        <View style={{ flex: 1 }} />
        {actionLabel && onActionPress ? (
          <Pressable
            onPress={onActionPress}
            hitSlop={8}
            style={({ pressed }) => [
              styles.action,
              pressed && { opacity: 0.65 },
            ]}
          >
            <Text style={[styles.actionText, { color: accent }]}>{actionLabel}</Text>
            <Ionicons name="chevron-forward" size={14} color={accent} />
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
    gap: 8,
  },
  bar: {
    width: 3,
    height: 14,
    borderRadius: 2,
  },
  label: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 12,
    letterSpacing: 1.0,
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
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.2,
  },
  caption: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    marginTop: 4,
    marginLeft: 11,
  },
});
