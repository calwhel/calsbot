import React from 'react';
import { View, Text, StyleSheet, Pressable, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';

import { colors, font, glow, radius, spacing } from '@/constants/colors';

type Step = {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  iconColor: string;
  title: string;
  body: string;
};

const STEPS: Step[] = [
  {
    icon: 'construct-outline',
    iconColor: '#22d3ee',
    title: 'Build a strategy',
    body: "Pick your coins, a trigger condition, and your risk. Takes about 60 seconds.",
  },
  {
    icon: 'pulse-outline',
    iconColor: '#a78bfa',
    title: 'We watch the markets',
    body: "Our engine scans 24/7 — you don't need to keep the app open.",
  },
  {
    icon: 'notifications-outline',
    iconColor: '#fbbf24',
    title: 'Get notified instantly',
    body: 'Push notifications + Telegram alerts fire the moment a trade opens or closes.',
  },
];

/**
 * First-run welcome card shown on Home when the user has zero strategies. Far
 * more inviting than a generic empty-state — it explains TradeHub in three
 * sentences and gives an obvious next-step CTA.
 */
export function QuickstartCard({ onStart }: { onStart: () => void }) {
  const uid = React.useId().replace(/:/g, '');
  const bgId = `qs-bg-${uid}`;
  const shineId = `qs-shine-${uid}`;
  const ctaId = `qs-cta-${uid}`;

  const handlePress = () => {
    if (Platform.OS !== 'web') {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => undefined);
    }
    onStart();
  };

  return (
    <View style={[styles.wrap, glow.accent]}>
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <SvgLinearGradient id={bgId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#1c2654" />
            <Stop offset="0.55" stopColor="#141c3b" />
            <Stop offset="1" stopColor="#0b1226" />
          </SvgLinearGradient>
          <SvgLinearGradient id={shineId} x1="0" y1="0" x2="1" y2="0">
            <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.7" />
            <Stop offset="0.6" stopColor="#a78bfa" stopOpacity="0.4" />
            <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill={`url(#${bgId})`} />
        <Rect width="100%" height="3" fill={`url(#${shineId})`} />
      </Svg>

      <View style={styles.inner}>
        <View style={styles.eyebrowRow}>
          <View style={styles.eyebrow}>
            <Ionicons name="sparkles" size={11} color={colors.accent} />
            <Text style={styles.eyebrowText}>WELCOME TO TRADEHUB</Text>
          </View>
        </View>
        <Text style={styles.title}>Automate any crypto strategy in three steps.</Text>
        <Text style={styles.subtitle}>
          You describe the rules — we watch the markets and tell you the moment something happens.
        </Text>

        <View style={{ marginTop: spacing.lg }}>
          {STEPS.map((step, i) => (
            <View key={`q-${i}`} style={styles.stepRow}>
              <View style={[styles.stepBubble, { backgroundColor: `${step.iconColor}24`, borderColor: `${step.iconColor}55` }]}>
                <Ionicons name={step.icon} size={18} color={step.iconColor} />
              </View>
              <View style={{ flex: 1, paddingLeft: 12 }}>
                <Text style={styles.stepTitle}>
                  <Text style={[styles.stepNum, { color: step.iconColor }]}>{i + 1}. </Text>
                  {step.title}
                </Text>
                <Text style={styles.stepBody}>{step.body}</Text>
              </View>
            </View>
          ))}
        </View>

        <Pressable
          onPress={handlePress}
          style={({ pressed }) => [
            styles.cta,
            pressed && { transform: [{ scale: 0.985 }], opacity: 0.92 },
          ]}
        >
          <Svg style={StyleSheet.absoluteFill}>
            <Defs>
              <SvgLinearGradient id={ctaId} x1="0" y1="0" x2="1" y2="1">
                <Stop offset="0" stopColor="#22d3ee" />
                <Stop offset="0.55" stopColor="#0ea5e9" />
                <Stop offset="1" stopColor="#6366f1" />
              </SvgLinearGradient>
            </Defs>
            <Rect width="100%" height="100%" fill={`url(#${ctaId})`} />
          </Svg>
          <Ionicons name="rocket-outline" size={18} color="#fff" />
          <Text style={styles.ctaText}>Build your first strategy</Text>
          <Ionicons name="arrow-forward" size={16} color="#fff" />
        </Pressable>

        <Text style={styles.fineprint}>
          No exchange account needed — strategies start in safe paper-trading mode.
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.18)',
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  inner: { padding: spacing.xl },
  eyebrowRow: { flexDirection: 'row' },
  eyebrow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(34,211,238,0.14)',
    borderColor: 'rgba(34,211,238,0.32)',
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
  },
  eyebrowText: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.7,
  },
  title: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 22,
    letterSpacing: -0.4,
    marginTop: spacing.md,
    lineHeight: 27,
  },
  subtitle: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13.5,
    marginTop: 6,
    lineHeight: 19,
  },
  stepRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 8,
  },
  stepBubble: {
    width: 34,
    height: 34,
    borderRadius: 11,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
  },
  stepTitle: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 14,
  },
  stepNum: { fontFamily: font.black },
  stepBody: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 12.5,
    marginTop: 2,
    lineHeight: 17,
  },
  cta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingHorizontal: spacing.lg,
    paddingVertical: 14,
    borderRadius: radius.pill,
    overflow: 'hidden',
    marginTop: spacing.lg,
  },
  ctaText: {
    color: '#fff',
    fontFamily: font.bold,
    fontSize: 15,
    letterSpacing: 0.2,
  },
  fineprint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    textAlign: 'center',
    marginTop: spacing.md,
    lineHeight: 15,
  },
});
