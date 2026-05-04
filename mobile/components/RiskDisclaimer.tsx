import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';

export function RiskDisclaimer({ compact }: { compact?: boolean }) {
  return (
    <View style={[styles.container, compact && styles.compact]}>
      <Ionicons name="warning-outline" size={compact ? 12 : 14} color={colors.textMute} />
      <Text style={[styles.text, compact && styles.textCompact]}>
        Trading involves significant risk. Past performance does not guarantee future results. You may lose some or all of your invested capital. Only trade with money you can afford to lose.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    backgroundColor: 'rgba(214,163,92,0.06)',
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: 'rgba(214,163,92,0.12)',
    marginTop: spacing.md,
  },
  compact: {
    paddingVertical: 6,
    paddingHorizontal: 10,
    marginTop: spacing.sm,
  },
  text: {
    flex: 1,
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    lineHeight: 15,
  },
  textCompact: {
    fontSize: 10,
    lineHeight: 13,
  },
});
