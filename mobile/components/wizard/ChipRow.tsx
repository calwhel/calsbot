import React from 'react';
import { View, Text, Pressable, StyleSheet, ScrollView, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';

export type ChipOption<T extends string> = {
  value: T;
  label: string;
  hint?: string;
  icon?: string;
};

/**
 * Horizontal pill row used everywhere in the wizard for picking a single value
 * from a small set (timeframe, direction, signal frequency, etc).
 *
 * Single-select by default; pass `multi` to allow toggling multiple values.
 */
export function ChipRow<T extends string>({
  options,
  value,
  onChange,
  multi = false,
  scroll = false,
  size = 'md',
}: {
  options: ChipOption<T>[];
  /** Active value(s). For multi-select, pass an array. */
  value: T | T[] | null;
  onChange: (next: T) => void;
  multi?: boolean;
  /** Wrap chips onto multiple rows by default; pass scroll=true for an h-scroll. */
  scroll?: boolean;
  size?: 'sm' | 'md';
}) {
  const isActive = (v: T) => Array.isArray(value) ? value.includes(v) : value === v;

  const handlePress = (v: T) => {
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    onChange(v);
  };

  const Wrap = scroll ? ScrollView : View;
  const wrapProps = scroll
    ? { horizontal: true, showsHorizontalScrollIndicator: false, contentContainerStyle: styles.scrollInner }
    : { style: styles.wrap };

  return (
    <Wrap {...(wrapProps as any)}>
      {options.map(opt => {
        const active = isActive(opt.value);
        return (
          <Pressable
            key={opt.value}
            onPress={() => handlePress(opt.value)}
            accessibilityRole={multi ? 'checkbox' : 'radio'}
            accessibilityState={{ selected: active }}
            style={({ pressed }) => [
              styles.chip,
              size === 'sm' && styles.chipSm,
              active && styles.chipActive,
              pressed && { opacity: 0.85 },
            ]}
          >
            {opt.icon ? (
              <Text style={[styles.chipIcon, size === 'sm' && { fontSize: 11 }]}>{opt.icon}</Text>
            ) : null}
            <Text
              style={[
                styles.chipLabel,
                size === 'sm' && styles.chipLabelSm,
                active && styles.chipLabelActive,
              ]}
              numberOfLines={1}
            >
              {opt.label}
            </Text>
            {opt.hint ? (
              <Text style={[styles.chipHint, active && styles.chipHintActive]}>
                {opt.hint}
              </Text>
            ) : null}
          </Pressable>
        );
      })}
    </Wrap>
  );
}

const styles = StyleSheet.create({
  wrap:       { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  scrollInner:{ flexDirection: 'row', gap: 6, paddingRight: spacing.lg },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
  },
  chipSm:        { paddingVertical: 5, paddingHorizontal: 9 },
  chipActive:    { borderColor: colors.accent, backgroundColor: colors.accentDim },
  chipIcon:      { fontSize: 13 },
  chipLabel:     { fontFamily: font.medium, fontSize: 12.5, color: colors.text },
  chipLabelSm:   { fontSize: 11.5 },
  chipLabelActive:{ color: colors.accent },
  chipHint:      { fontFamily: font.regular, fontSize: 10.5, color: colors.textMute },
  chipHintActive:{ color: colors.accent, opacity: 0.85 },
});
