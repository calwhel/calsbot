import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * Compact, neutral info-card primitive used across the strategy detail screen
 * and the home quickstart. Renders as a bordered card with a section label and
 * children stacked inside a tinted block. Optional eyebrow icon adds a small
 * coloured chip to the title row.
 */
export function InfoCard({
  label,
  icon,
  iconColor,
  children,
}: {
  label: string;
  icon?: React.ComponentProps<typeof Ionicons>['name'];
  iconColor?: string;
  children: React.ReactNode;
}) {
  return (
    <View style={styles.card}>
      <View style={styles.header}>
        {icon ? (
          <View style={[styles.iconChip, iconColor ? { backgroundColor: `${iconColor}1f`, borderColor: `${iconColor}3a` } : null]}>
            <Ionicons name={icon} size={13} color={iconColor || colors.accent} />
          </View>
        ) : null}
        <Text style={styles.label}>{label}</Text>
      </View>
      <View style={styles.body}>{children}</View>
    </View>
  );
}

/** Single labelled row inside an InfoCard. Shows a label + value, with an
 *  optional secondary line beneath. */
export function InfoRow({
  label,
  value,
  hint,
  valueColor,
  mono,
}: {
  label: string;
  value: string;
  hint?: string;
  valueColor?: string;
  mono?: boolean;
}) {
  return (
    <View style={styles.row}>
      <View style={{ flex: 1 }}>
        <Text style={styles.rowLabel}>{label}</Text>
        {hint ? <Text style={styles.rowHint}>{hint}</Text> : null}
      </View>
      <Text
        style={[
          styles.rowValue,
          { color: valueColor || colors.text },
          mono && { fontVariant: ['tabular-nums'] },
        ]}
      >
        {value}
      </Text>
    </View>
  );
}

/** Bullet line — used for plain-English condition lists. */
export function InfoBullet({
  text,
  icon = 'ellipse',
  iconColor,
}: {
  text: string;
  icon?: React.ComponentProps<typeof Ionicons>['name'];
  iconColor?: string;
}) {
  return (
    <View style={styles.bullet}>
      <Ionicons
        name={icon}
        size={icon === 'ellipse' ? 6 : 14}
        color={iconColor || colors.accent}
        style={{ marginTop: icon === 'ellipse' ? 7 : 2, marginRight: 10 }}
      />
      <Text style={styles.bulletText}>{text}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    gap: 8,
    borderBottomWidth: 1,
    borderBottomColor: colors.divider,
  },
  iconChip: {
    width: 22,
    height: 22,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
  },
  body: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: colors.divider,
  },
  rowLabel: {
    color: colors.text,
    fontFamily: font.medium,
    fontSize: 13,
  },
  rowHint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    marginTop: 2,
  },
  rowValue: {
    fontFamily: font.bold,
    fontSize: 14,
  },
  bullet: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 6,
  },
  bulletText: {
    flex: 1,
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 19,
  },
});
