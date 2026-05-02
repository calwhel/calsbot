import React from 'react';
import { View, Text, Pressable, StyleSheet, Platform } from 'react-native';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';
import { ChipRow, type ChipOption } from './ChipRow';

/**
 * Touch-friendly numeric input used everywhere risk/exit values are picked.
 *
 *   [Take Profit]                    [-]   3.0%   [+]
 *   Quick presets: 1.5  3  5  8  12
 *
 * - `presets` are tappable shortcut chips that snap to a value.
 * - `step` is the +/- delta; tap-and-hold not supported (rarely needed).
 * - Value is clamped to [min, max].
 */
export function Stepper({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  unit = '',
  presets,
  hint,
  decimals = 0,
  disabled = false,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  presets?: number[];
  hint?: string;
  decimals?: number;
  disabled?: boolean;
}) {
  const clamp = (n: number) => Math.max(min, Math.min(max, n));
  const fmt = (n: number) => decimals > 0 ? n.toFixed(decimals) : String(Math.round(n));

  const bump = (delta: number) => {
    if (disabled) return;
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    const next = +clamp(value + delta).toFixed(decimals);
    onChange(next);
  };

  const presetOpts: ChipOption<string>[] = (presets || []).map(p => ({
    value: String(p),
    label: `${fmt(p)}${unit}`,
  }));

  return (
    <View style={[styles.wrap, disabled && { opacity: 0.55 }]}>
      <View style={styles.headerRow}>
        <Text style={styles.label}>{label}</Text>
        <View style={styles.controlRow}>
          <Pressable
            onPress={() => bump(-step)}
            disabled={disabled || value <= min}
            style={({ pressed }) => [
              styles.btn,
              (disabled || value <= min) && styles.btnDisabled,
              pressed && { opacity: 0.7 },
            ]}
            accessibilityRole="button"
            accessibilityLabel={`Decrease ${label}`}
          >
            <Text style={styles.btnTxt}>−</Text>
          </Pressable>
          <View style={styles.valueBox}>
            <Text style={styles.value}>{fmt(value)}{unit}</Text>
          </View>
          <Pressable
            onPress={() => bump(step)}
            disabled={disabled || value >= max}
            style={({ pressed }) => [
              styles.btn,
              (disabled || value >= max) && styles.btnDisabled,
              pressed && { opacity: 0.7 },
            ]}
            accessibilityRole="button"
            accessibilityLabel={`Increase ${label}`}
          >
            <Text style={styles.btnTxt}>+</Text>
          </Pressable>
        </View>
      </View>
      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
      {presets && presets.length ? (
        <View style={{ marginTop: 6 }}>
          <ChipRow
            options={presetOpts}
            value={String(value)}
            onChange={(v) => {
              if (disabled) return;
              const n = Number(v);
              if (!Number.isNaN(n)) onChange(clamp(n));
            }}
            size="sm"
          />
        </View>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: spacing.md },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  label: { fontFamily: font.semibold, fontSize: 13, color: colors.text, flexShrink: 1, paddingRight: 8 },
  controlRow: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  btn: {
    width: 36, height: 36,
    borderRadius: radius.md,
    backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
    alignItems: 'center', justifyContent: 'center',
  },
  btnDisabled: { opacity: 0.4 },
  btnTxt: {
    fontFamily: font.bold,
    fontSize: 22,
    color: colors.text,
    lineHeight: 24,
    marginTop: -2,
  },
  valueBox: {
    minWidth: 78,
    height: 36,
    borderRadius: radius.md,
    backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.borderHi,
    alignItems: 'center', justifyContent: 'center',
    paddingHorizontal: 10,
  },
  value: { fontFamily: font.bold, fontSize: 14, color: colors.text },
  hint:  { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute, marginTop: 4, lineHeight: 16 },
});
