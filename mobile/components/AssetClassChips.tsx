import React from 'react';
import { Pressable, ScrollView, StyleSheet, Text } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { colors, radius, spacing } from '@/constants/colors';

export type AssetClassKey = 'all' | 'crypto' | 'stock' | 'forex' | 'index';

const CLASSES: { key: AssetClassKey; label: string; icon: keyof typeof Ionicons.glyphMap }[] = [
  { key: 'all',    label: 'All',     icon: 'grid-outline' },
  { key: 'crypto', label: 'Crypto',  icon: 'logo-bitcoin' },
  { key: 'stock',  label: 'Stocks',  icon: 'business-outline' },
  { key: 'forex',  label: 'Forex',   icon: 'swap-horizontal' },
  { key: 'index',  label: 'Indices', icon: 'stats-chart-outline' },
];

export function AssetClassChips({
  value,
  counts,
  onChange,
}: {
  value: AssetClassKey;
  counts?: Record<string, number>;
  onChange: (v: AssetClassKey) => void;
}) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.row}
    >
      {CLASSES.map((c) => {
        const active = c.key === value;
        const n = counts ? counts[c.key] ?? 0 : null;
        const disabled = counts ? c.key !== 'all' && n === 0 : false;
        return (
          <Pressable
            key={c.key}
            disabled={disabled}
            onPress={() => {
              if (active || disabled) return;
              Haptics.selectionAsync().catch(() => {});
              onChange(c.key);
            }}
            style={({ pressed }) => [
              styles.chip,
              active && styles.chipActive,
              disabled && { opacity: 0.4 },
              pressed && !disabled && { opacity: 0.85 },
            ]}
          >
            <Ionicons
              name={c.icon}
              size={12}
              color={active ? colors.accentText : colors.textDim}
            />
            <Text style={[styles.text, active && styles.textActive]}>{c.label}</Text>
            {n !== null && n > 0 && (
              <Text style={[styles.count, active && styles.countActive]}>{n}</Text>
            )}
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  row: {
    paddingHorizontal: spacing.lg,
    gap: 8,
    paddingTop: spacing.sm,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: radius.pill,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  text: {
    fontSize: 12,
    fontWeight: '600',
    color: colors.textDim,
  },
  textActive: { color: colors.accentText },
  count: {
    fontSize: 10,
    fontWeight: '700',
    color: colors.textDim,
    marginLeft: 2,
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 6,
    backgroundColor: colors.bg,
    overflow: 'hidden',
  },
  countActive: {
    color: colors.accent,
    backgroundColor: 'rgba(255,255,255,0.92)',
  },
});
