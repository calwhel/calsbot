import React from 'react';
import { Pressable, StyleSheet, Text, View, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';
import { colors, font, radius, spacing } from '@/constants/colors';

type Tone = 'accent' | 'positive' | 'violet' | 'gold' | 'magenta' | 'mint';

const PALETTE: Record<Tone, [string, string]> = {
  accent:   ['#22d3ee', '#3b82f6'],
  positive: ['#34d399', '#10b981'],
  violet:   ['#a78bfa', '#7c3aed'],
  gold:     ['#fbbf24', '#f5b754'],
  magenta:  ['#f472b6', '#db2777'],
  mint:     ['#5eead4', '#22d3ee'],
};

/**
 * ActionTile — gradient-filled icon button used in the Home quick-actions row.
 *
 *   ┌────────────┐
 *   │     ◇      │  <- gradient circle w/ icon
 *   │  Run scan  │
 *   └────────────┘
 *
 * Compact (~80px wide) tiles laid out horizontally. Triggers a haptic
 * on press. Use for top-level actions a user reaches for often.
 */
export function ActionTile({
  icon,
  label,
  tone = 'accent',
  onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  tone?: Tone;
  onPress: () => void;
}) {
  const uid = React.useId().replace(/:/g, '');
  const ringId = `act-${uid}`;
  const [c0, c1] = PALETTE[tone];

  const handle = () => {
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    onPress();
  };

  return (
    <Pressable
      onPress={handle}
      style={({ pressed }) => [
        styles.tile,
        pressed && { opacity: 0.85, transform: [{ scale: 0.96 }] },
      ]}
    >
      <View style={styles.iconWrap}>
        <Svg width={48} height={48}>
          <Defs>
            <SvgLinearGradient id={ringId} x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0" stopColor={c0} />
              <Stop offset="1" stopColor={c1} />
            </SvgLinearGradient>
          </Defs>
          <Rect width="48" height="48" rx="14" ry="14" fill={`url(#${ringId})`} />
        </Svg>
        <View style={styles.iconCenter} pointerEvents="none">
          <Ionicons name={icon} size={22} color="#0a1024" />
        </View>
      </View>
      <Text style={styles.label} numberOfLines={1}>{label}</Text>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  tile: {
    width: 78,
    alignItems: 'center',
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  iconWrap: {
    width: 48,
    height: 48,
    borderRadius: 14,
    overflow: 'hidden',
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconCenter: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.1,
    textAlign: 'center',
  },
  _: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.md,
  },
});
