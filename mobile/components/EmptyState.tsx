import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, spacing } from '@/constants/colors';

export function EmptyState({
  icon = 'cube-outline',
  title,
  hint,
}: {
  icon?: keyof typeof Ionicons.glyphMap;
  title: string;
  hint?: string;
}) {
  return (
    <View style={styles.wrap}>
      <Ionicons name={icon} size={42} color={colors.textMute} />
      <Text style={styles.title}>{title}</Text>
      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
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
  title: {
    color: colors.text,
    fontSize: 16,
    fontWeight: '600',
    marginTop: spacing.md,
    textAlign: 'center',
  },
  hint: {
    color: colors.textMute,
    fontSize: 13,
    marginTop: spacing.xs,
    textAlign: 'center',
    lineHeight: 18,
  },
});
