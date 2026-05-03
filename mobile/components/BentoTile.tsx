import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';

type Tone = 'accent' | 'positive' | 'negative' | 'warning' | 'violet' | 'gold' | 'mint' | 'magenta' | 'indigo' | 'neutral';

/**
 * In modern-dark we collapse all decorative tones to two: positive (green)
 * and negative (red). Everything else renders neutral. The label/value/sub
 * structure is unchanged so existing call-sites continue to work.
 */
function valueColorFor(tone: Tone): string {
  // Only explicit performance tones ever colour the number. 'accent' is an
  // action/neutral signal in modern-dark, so it must NOT imply gain.
  if (tone === 'positive' || tone === 'mint') return colors.positive;
  if (tone === 'negative') return colors.negative;
  if (tone === 'warning' || tone === 'gold') return colors.warning;
  return colors.text;
}
function iconColorFor(tone: Tone): string {
  // Icons stay neutral except for explicit positive/negative — keeps the
  // grid quiet rather than a rainbow of chips.
  if (tone === 'positive' || tone === 'mint') return colors.positive;
  if (tone === 'negative') return colors.negative;
  return colors.textDim;
}

export function BentoTile({
  icon,
  label,
  value,
  sub,
  tone = 'neutral',
  onPress,
  size = 'md',
}: {
  icon?: keyof typeof Ionicons.glyphMap;
  label: string;
  value: string;
  sub?: string;
  tone?: Tone;
  onPress?: () => void;
  size?: 'sm' | 'md' | 'lg';
}) {
  const valueColor = valueColorFor(tone);
  const iconColor = iconColorFor(tone);

  const Container: any = onPress ? Pressable : View;
  const containerProps = onPress
    ? {
        onPress,
        style: ({ pressed }: { pressed: boolean }) => [
          styles.tile,
          size === 'sm' && styles.tileSm,
          size === 'lg' && styles.tileLg,
          pressed && { opacity: 0.85 },
        ],
      }
    : {
        style: [
          styles.tile,
          size === 'sm' && styles.tileSm,
          size === 'lg' && styles.tileLg,
        ],
      };

  return (
    <Container {...containerProps}>
      {icon ? (
        <View style={styles.iconChip}>
          <Ionicons name={icon} size={14} color={iconColor} />
        </View>
      ) : null}

      <Text style={styles.label} numberOfLines={1}>{label}</Text>
      <Text
        style={[
          styles.value,
          { color: valueColor },
          size === 'sm' && styles.valueSm,
          size === 'lg' && styles.valueLg,
        ]}
        numberOfLines={1}
      >
        {value}
      </Text>
      {sub ? <Text style={styles.sub} numberOfLines={1}>{sub}</Text> : null}
    </Container>
  );
}

const styles = StyleSheet.create({
  tile: {
    flex: 1,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    minHeight: 104,
    justifyContent: 'space-between',
  },
  tileSm: { minHeight: 86, padding: spacing.md },
  tileLg: { minHeight: 132, padding: spacing.lg },
  iconChip: {
    width: 24,
    height: 24,
    borderRadius: 6,
    backgroundColor: colors.cardHi,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 11,
    letterSpacing: 0.4,
    textTransform: 'uppercase',
    marginTop: spacing.xs,
  },
  value: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 22,
    letterSpacing: -0.5,
    fontVariant: ['tabular-nums'],
    marginTop: 4,
  },
  valueSm: { fontSize: 18 },
  valueLg: { fontSize: 28, letterSpacing: -0.8 },
  sub: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    marginTop: 4,
  },
});
