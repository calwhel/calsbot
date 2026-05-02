import React from 'react';
import { View, Text, StyleSheet, Pressable } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Defs, LinearGradient as SvgLinearGradient, RadialGradient, Stop, Rect } from 'react-native-svg';
import { colors, font, radius, spacing } from '@/constants/colors';

type Tone = 'accent' | 'positive' | 'negative' | 'warning' | 'violet' | 'gold' | 'mint' | 'magenta' | 'indigo' | 'neutral';

const TONE_PALETTE: Record<Tone, { primary: string; soft: string; glow: string }> = {
  accent:   { primary: '#22d3ee', soft: 'rgba(34,211,238,0.22)',  glow: 'rgba(34,211,238,0.35)' },
  positive: { primary: '#34d399', soft: 'rgba(52,211,153,0.22)',  glow: 'rgba(52,211,153,0.30)' },
  negative: { primary: '#f87171', soft: 'rgba(248,113,113,0.22)', glow: 'rgba(248,113,113,0.30)' },
  warning:  { primary: '#fbbf24', soft: 'rgba(251,191,36,0.22)',  glow: 'rgba(251,191,36,0.30)' },
  violet:   { primary: '#a78bfa', soft: 'rgba(167,139,250,0.22)', glow: 'rgba(167,139,250,0.30)' },
  gold:     { primary: '#f5b754', soft: 'rgba(245,183,84,0.22)',  glow: 'rgba(245,183,84,0.30)' },
  mint:     { primary: '#5eead4', soft: 'rgba(94,234,212,0.22)',  glow: 'rgba(94,234,212,0.30)' },
  magenta:  { primary: '#f472b6', soft: 'rgba(244,114,182,0.22)', glow: 'rgba(244,114,182,0.30)' },
  indigo:   { primary: '#6366f1', soft: 'rgba(99,102,241,0.24)',  glow: 'rgba(99,102,241,0.32)' },
  neutral:  { primary: '#a3acc7', soft: 'rgba(163,172,199,0.18)', glow: 'rgba(0,0,0,0)' },
};

/**
 * BentoTile — premium metric tile used in bento-grid layouts.
 *
 *   ┌────────────────────┐
 *   │  ◇ ICON            │
 *   │                    │
 *   │  WIN RATE          │
 *   │  68%               │
 *   │  124 closed trades │
 *   └────────────────────┘
 *
 * Has a tone-tinted radial halo top-left, a small icon chip, an uppercase
 * label, a big tabular value, and an optional sub-label. Press handler
 * makes the whole tile tappable with a subtle scale animation.
 */
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
  const palette = TONE_PALETTE[tone];
  const uid = React.useId().replace(/:/g, '');
  const haloId = `bt-halo-${uid}`;
  const sheenId = `bt-sheen-${uid}`;

  const Container: any = onPress ? Pressable : View;
  const containerProps = onPress
    ? {
        onPress,
        style: ({ pressed }: { pressed: boolean }) => [
          styles.tile,
          size === 'sm' && styles.tileSm,
          size === 'lg' && styles.tileLg,
          pressed && { opacity: 0.88, transform: [{ scale: 0.985 }] },
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
      <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <RadialGradient id={haloId} cx="0%" cy="0%" rx="80%" ry="80%">
            <Stop offset="0" stopColor={palette.primary} stopOpacity={tone === 'neutral' ? '0' : '0.16'} />
            <Stop offset="1" stopColor={palette.primary} stopOpacity="0" />
          </RadialGradient>
          <SvgLinearGradient id={sheenId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor="#1d2542" stopOpacity="1" />
            <Stop offset="1" stopColor="#101729" stopOpacity="1" />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100" height="100" fill={`url(#${sheenId})`} />
        <Rect width="100" height="100" fill={`url(#${haloId})`} />
      </Svg>

      {icon ? (
        <View
          style={[
            styles.iconChip,
            { backgroundColor: palette.soft, borderColor: palette.soft },
          ]}
        >
          <Ionicons name={icon} size={14} color={palette.primary} />
        </View>
      ) : null}

      <Text style={styles.label} numberOfLines={1}>{label}</Text>
      <Text
        style={[
          styles.value,
          tone !== 'neutral' && { color: palette.primary },
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
    backgroundColor: '#141a2c',
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    padding: spacing.lg,
    minHeight: 110,
    overflow: 'hidden',
    justifyContent: 'space-between',
  },
  tileSm: { minHeight: 90, padding: spacing.md },
  tileLg: { minHeight: 140, padding: spacing.lg },
  iconChip: {
    width: 28,
    height: 28,
    borderRadius: 8,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.sm,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 10.5,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginTop: spacing.xs,
  },
  value: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 24,
    letterSpacing: -0.6,
    fontVariant: ['tabular-nums'],
    marginTop: 4,
  },
  valueSm: { fontSize: 19 },
  valueLg: { fontSize: 30, letterSpacing: -1.0 },
  sub: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
    marginTop: 4,
  },
});
