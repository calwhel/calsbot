import React from 'react';
import { View, Text, StyleSheet, Pressable, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { colors, font, radius, spacing } from '@/constants/colors';

type Step = {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  title: string;
  body: string;
};

const STEPS: Step[] = [
  {
    icon: 'construct-outline',
    title: 'Build a strategy',
    body: "Pick your coins, a trigger condition, and your risk. Takes about 60 seconds.",
  },
  {
    icon: 'pulse-outline',
    title: 'We watch the markets',
    body: "Our engine scans 24/7 — you don't need to keep the app open.",
  },
  {
    icon: 'notifications-outline',
    title: 'Get notified instantly',
    body: 'Push notifications + Telegram alerts fire the moment a trade opens or closes.',
  },
];

/**
 * Modern-dark Quickstart — flat surface, neutral step bubbles, single solid
 * CTA. No gradients, no shines.
 */
export function QuickstartCard({ onStart }: { onStart: () => void }) {
  const handlePress = () => {
    if (Platform.OS !== 'web') {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => undefined);
    }
    onStart();
  };

  return (
    <View style={styles.wrap}>
      <View style={styles.inner}>
        <Text style={styles.eyebrow}>WELCOME TO TRADEHUB</Text>
        <Text style={styles.title}>Automate any crypto strategy in three steps.</Text>
        <Text style={styles.subtitle}>
          You describe the rules — we watch the markets and tell you the moment something happens.
        </Text>

        <View style={{ marginTop: spacing.lg }}>
          {STEPS.map((step, i) => (
            <View key={`q-${i}`} style={styles.stepRow}>
              <View style={styles.stepBubble}>
                <Ionicons name={step.icon} size={16} color={colors.textDim} />
              </View>
              <View style={{ flex: 1, paddingLeft: 12 }}>
                <Text style={styles.stepTitle}>
                  <Text style={styles.stepNum}>{i + 1}. </Text>
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
            pressed && { opacity: 0.85 },
          ]}
        >
          <Text style={styles.ctaText}>Build your first strategy</Text>
          <Ionicons name="arrow-forward" size={15} color={colors.bg} />
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
    borderColor: colors.border,
    backgroundColor: colors.card,
    overflow: 'hidden',
  },
  inner: { padding: spacing.xl },
  eyebrow: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 11,
    letterSpacing: 0.8,
  },
  title: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 22,
    letterSpacing: -0.4,
    marginTop: spacing.sm,
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
    width: 32,
    height: 32,
    borderRadius: radius.md,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepTitle: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 14,
  },
  stepNum: { color: colors.textDim, fontFamily: font.semibold },
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
    borderRadius: radius.md,
    backgroundColor: colors.text,
    marginTop: spacing.lg,
  },
  ctaText: {
    color: colors.bg,
    fontFamily: font.semibold,
    fontSize: 15,
    letterSpacing: 0.1,
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
