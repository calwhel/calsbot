import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { colors, font, spacing } from '@/constants/colors';

/** Compact uppercase header used above each control group inside a wizard
 *  step card. Smaller and tighter than the home-screen SectionLabel. */
export function SectionHeader({
  label,
  hint,
  icon,
}: {
  label: string;
  hint?: string;
  icon?: string;
}) {
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        {icon ? <Text style={styles.icon}>{icon}</Text> : null}
        <Text style={styles.label}>{label}</Text>
      </View>
      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { marginBottom: spacing.sm, marginTop: spacing.sm },
  row:  { flexDirection: 'row', alignItems: 'center', gap: 6 },
  icon: { fontSize: 13 },
  label:{
    fontFamily: font.bold,
    fontSize: 11,
    color: colors.textDim,
    letterSpacing: 1.2,
    textTransform: 'uppercase',
  },
  hint: {
    fontFamily: font.regular,
    fontSize: 12,
    color: colors.textMute,
    marginTop: 2,
    lineHeight: 16,
  },
});
