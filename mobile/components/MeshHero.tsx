import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Path } from 'react-native-svg';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * MeshHero (modern-dark) — flat hero surface for the Home P&L panel.
 *
 *   ┌──────────────────────────────────────────────────────┐
 *   │  TOTAL P&L                              ▲ +4.2% · 30d │
 *   │                                                       │
 *   │  +18.74%                                              │
 *   │                                                       │
 *   │  Across all strategies                                │
 *   │  ────╱╲────╱─╲──╱──╲──── (sparkline)                  │
 *   └──────────────────────────────────────────────────────┘
 *
 * All decorative orbs / mesh gradients / shines have been removed in
 * favour of a flat surface, hairline border, tight type, and a single
 * thin sparkline at the bottom. Same public API as the previous version.
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
  const sparkId = `mh-spark-${uid}`;

  const valueColor =
    tone === 'positive' ? colors.positive : tone === 'negative' ? colors.negative : colors.text;

  // Sparkline math
  const SPARK_W = 320;
  const SPARK_H = 44;
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
    tone === 'positive' ? colors.positive : tone === 'negative' ? colors.negative : colors.textDim;

  const badgeColor =
    badgeTone === 'positive'
      ? colors.positive
      : badgeTone === 'negative'
      ? colors.negative
      : colors.textDim;

  return (
    <View style={styles.hero}>
      <View style={styles.heroInner}>
        <View style={styles.heroLabelRow}>
          <Text style={styles.heroLabel}>{label}</Text>
          {badgeText ? (
            <View style={styles.heroBadge}>
              <Ionicons
                name={badgeTone === 'negative' ? 'trending-down' : 'trending-up'}
                size={11}
                color={badgeColor}
              />
              <Text style={[styles.heroBadgeText, { color: badgeColor }]}>{badgeText}</Text>
            </View>
          ) : null}
        </View>

        <Text style={[styles.heroValue, { color: valueColor }]} numberOfLines={1} adjustsFontSizeToFit>
          {value}
        </Text>

        {footnote ? <Text style={styles.heroFootnote}>{footnote}</Text> : null}

        {linePath ? (
          <View style={styles.sparkRow}>
            <Svg width="100%" height={SPARK_H} viewBox={`0 0 ${SPARK_W} ${SPARK_H}`} preserveAspectRatio="none">
              <Defs>
                <SvgLinearGradient id={sparkId} x1="0" y1="0" x2="0" y2="1">
                  <Stop offset="0" stopColor={sparkStroke} stopOpacity="0.18" />
                  <Stop offset="1" stopColor={sparkStroke} stopOpacity="0" />
                </SvgLinearGradient>
              </Defs>
              <Path d={fillPath} fill={`url(#${sparkId})`} />
              <Path
                d={linePath}
                fill="none"
                stroke={sparkStroke}
                strokeWidth={1.5}
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
    borderColor: colors.border,
    overflow: 'hidden',
    backgroundColor: colors.card,
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
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.8,
    textTransform: 'uppercase',
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'transparent',
    paddingHorizontal: 0,
    paddingVertical: 0,
  },
  heroBadgeText: {
    fontFamily: font.semibold,
    fontSize: 12,
    letterSpacing: 0,
    fontVariant: ['tabular-nums'],
  },
  heroValue: {
    fontFamily: font.semibold,
    fontSize: 44,
    letterSpacing: -1.2,
    marginTop: spacing.md,
    fontVariant: ['tabular-nums'],
    lineHeight: 50,
  },
  heroFootnote: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    marginTop: 6,
    lineHeight: 18,
  },
  sparkRow: {
    marginTop: spacing.lg,
    marginHorizontal: -spacing.xl,
    height: 44,
    overflow: 'hidden',
  },
});
