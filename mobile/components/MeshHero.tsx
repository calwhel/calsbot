import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  RadialGradient,
  Stop,
  Rect,
  Circle,
  Path,
} from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, glow, radius, spacing } from '@/constants/colors';

/**
 * MeshHero — premium hero card for the Home P&L panel.
 *
 *   ┌──────────────────────────────────────────────────────┐
 *   │ TOTAL P&L                            ▲ +4.2% · 30d   │
 *   │                                                       │
 *   │ +18.74%                                               │
 *   │                                       ╭──────╮        │
 *   │ Across all strategies            ╭───╯      ╰──╮     │
 *   │                              ───╯              ╰──   │
 *   └──────────────────────────────────────────────────────┘
 *
 * Multi-stop mesh-style gradient backdrop, glowing orb top-right,
 * soft top shine line, and an inline sparkline overlay.
 */
export function MeshHero({
  label,
  value,
  badgeText,
  badgeTone = 'positive',
  footnote,
  spark,
  tone = 'positive',
}: {
  label: string;
  value: string;
  badgeText?: string;
  badgeTone?: 'positive' | 'negative' | 'neutral';
  footnote?: string;
  spark?: number[];
  tone?: 'positive' | 'negative' | 'neutral';
}) {
  const uid = React.useId().replace(/:/g, '');
  const bgId = `mh-bg-${uid}`;
  const orbId = `mh-orb-${uid}`;
  const orb2Id = `mh-orb2-${uid}`;
  const shineId = `mh-shine-${uid}`;
  const sparkId = `mh-spark-${uid}`;

  const valueColor =
    tone === 'positive' ? colors.positive : tone === 'negative' ? colors.negative : colors.text;

  // Sparkline math (drawn at the bottom of the card if `spark` provided)
  const SPARK_W = 320;
  const SPARK_H = 56;
  let linePath = '';
  let fillPath = '';
  if (spark && spark.length >= 2) {
    const min = Math.min(...spark);
    const max = Math.max(...spark);
    const range = max - min || 1;
    const dx = SPARK_W / (spark.length - 1);
    const pts = spark.map((v, i) => ({
      x: i * dx,
      y: SPARK_H - ((v - min) / range) * (SPARK_H - 4) - 2,
    }));
    linePath = pts.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ');
    fillPath = `${linePath} L ${pts[pts.length - 1].x} ${SPARK_H} L ${pts[0].x} ${SPARK_H} Z`;
  }

  const sparkStroke =
    tone === 'positive' ? '#34d399' : tone === 'negative' ? '#f87171' : '#67e8f9';

  return (
    <View style={[styles.hero, glow.accent]}>
      <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <SvgLinearGradient id={bgId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#1f2a5e" />
            <Stop offset="0.4" stopColor="#161b3d" />
            <Stop offset="0.75" stopColor="#0e1330" />
            <Stop offset="1" stopColor="#08091e" />
          </SvgLinearGradient>
          <RadialGradient id={orbId} cx="85%" cy="20%" rx="55%" ry="55%">
            <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.55" />
            <Stop offset="0.5" stopColor="#3b82f6" stopOpacity="0.18" />
            <Stop offset="1" stopColor="#3b82f6" stopOpacity="0" />
          </RadialGradient>
          <RadialGradient id={orb2Id} cx="10%" cy="95%" rx="50%" ry="55%">
            <Stop offset="0" stopColor="#a78bfa" stopOpacity="0.32" />
            <Stop offset="1" stopColor="#a78bfa" stopOpacity="0" />
          </RadialGradient>
          <SvgLinearGradient id={shineId} x1="0" y1="0" x2="1" y2="0">
            <Stop offset="0" stopColor="#22d3ee" stopOpacity="0" />
            <Stop offset="0.5" stopColor="#67e8f9" stopOpacity="0.95" />
            <Stop offset="1" stopColor="#a78bfa" stopOpacity="0" />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100" height="100" fill={`url(#${bgId})`} />
        <Rect width="100" height="100" fill={`url(#${orbId})`} />
        <Rect width="100" height="100" fill={`url(#${orb2Id})`} />
        <Rect width="100" height="0.4" fill={`url(#${shineId})`} />
      </Svg>

      {/* Decorative orb circles */}
      <View pointerEvents="none" style={styles.orbWrap}>
        <Svg width={170} height={170}>
          <Defs>
            <RadialGradient id={`mh-cg-${uid}`} cx="50%" cy="50%" rx="50%" ry="50%">
              <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.18" />
              <Stop offset="0.7" stopColor="#22d3ee" stopOpacity="0.04" />
              <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Circle cx="85" cy="85" r="80" fill={`url(#mh-cg-${uid})`} />
          <Circle cx="85" cy="85" r="55" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
          <Circle cx="85" cy="85" r="38" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
        </Svg>
      </View>

      <View style={styles.heroInner}>
        <View style={styles.heroLabelRow}>
          <Text style={styles.heroLabel}>{label}</Text>
          {badgeText ? (
            <View style={styles.heroBadge}>
              <Ionicons
                name={badgeTone === 'negative' ? 'trending-down' : 'trending-up'}
                size={12}
                color={
                  badgeTone === 'positive'
                    ? colors.positive
                    : badgeTone === 'negative'
                    ? colors.negative
                    : colors.textDim
                }
              />
              <Text
                style={[
                  styles.heroBadgeText,
                  {
                    color:
                      badgeTone === 'positive'
                        ? colors.positive
                        : badgeTone === 'negative'
                        ? colors.negative
                        : colors.textDim,
                  },
                ]}
              >
                {badgeText}
              </Text>
            </View>
          ) : null}
        </View>

        <Text style={[styles.heroValue, { color: valueColor }]} numberOfLines={1} adjustsFontSizeToFit>
          {value}
        </Text>

        {footnote ? <Text style={styles.heroFootnote}>{footnote}</Text> : null}

        {/* Inline sparkline strip at the bottom */}
        {linePath ? (
          <View style={styles.sparkRow}>
            <Svg width="100%" height={SPARK_H} viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} preserveAspectRatio="none">
              <Defs>
                <SvgLinearGradient id={sparkId} x1="0" y1="0" x2="0" y2="1">
                  <Stop offset="0" stopColor={sparkStroke} stopOpacity="0.45" />
                  <Stop offset="1" stopColor={sparkStroke} stopOpacity="0" />
                </SvgLinearGradient>
              </Defs>
              <Path d={fillPath} fill={`url(#${sparkId})`} />
              <Path
                d={linePath}
                fill="none"
                stroke={sparkStroke}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
            </Svg>
          </View>
        ) : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  hero: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.22)',
    overflow: 'hidden',
    backgroundColor: '#0a1024',
    minHeight: 220,
  },
  orbWrap: {
    position: 'absolute',
    top: -30,
    right: -30,
    opacity: 0.85,
  },
  heroInner: {
    padding: spacing.xl,
    paddingBottom: spacing.lg,
  },
  heroLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  heroLabel: {
    color: '#9aa6c7',
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 1.2,
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(0,0,0,0.38)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.07)',
  },
  heroBadgeText: {
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.3,
  },
  heroValue: {
    fontFamily: font.black,
    fontSize: 60,
    letterSpacing: -2.0,
    marginTop: 14,
    fontVariant: ['tabular-nums'],
    lineHeight: 64,
  },
  heroFootnote: {
    color: '#a3acc7',
    fontFamily: font.medium,
    fontSize: 13,
    marginTop: spacing.sm,
    lineHeight: 18,
  },
  sparkRow: {
    marginTop: spacing.lg,
    marginHorizontal: -spacing.xl,
    height: 56,
    overflow: 'hidden',
  },
});
