import React, { useEffect, useRef, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  Pressable,
  Animated,
  Easing,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as SecureStore from 'expo-secure-store';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, shadow, spacing } from '@/constants/colors';

const STORAGE_KEY = 'tradehub_onboarded_v1';

type Step = {
  icon: keyof typeof Ionicons.glyphMap;
  title: string;
  body: string;
};

const STEPS: Step[] = [
  {
    icon: 'rocket-outline',
    title: 'Welcome to TradeHub',
    body: 'Build trading strategies, run them on auto-pilot, and track every fire from your phone. A quick tour — under 30 seconds.',
  },
  {
    icon: 'analytics-outline',
    title: 'Your P&L at a glance',
    body: 'The Home tab shows your total P&L, equity curve, and recent trades. Pull down to refresh anytime.',
  },
  {
    icon: 'pulse-outline',
    title: 'Strategies, your way',
    body: 'The Strategies tab lets you build with the wizard, with AI chat, or paste a Pine script. Toggle paper / live trading per strategy.',
  },
  {
    icon: 'storefront-outline',
    title: 'Discover in the Market',
    body: 'Browse strategies built by other traders, see real performance, and copy the ones you like with one tap.',
  },
  {
    icon: 'shield-checkmark-outline',
    title: 'Connect your exchange',
    body: 'When you\'re ready to go live, head to Account → Exchanges to connect Bitunix. Until then, every strategy runs in paper mode.',
  },
];

/**
 * OnboardingTour — full-screen modal that walks new users through the app.
 * Persists completion in SecureStore so it only shows once. Skippable at
 * any step. Renders nothing until the first-launch check resolves.
 */
export function OnboardingTour() {
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);
  const opacity = useRef(new Animated.Value(0)).current;
  const slide = useRef(new Animated.Value(20)).current;

  // Check first-launch flag once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const seen = Platform.OS === 'web'
          ? (typeof window !== 'undefined' ? window.localStorage.getItem(STORAGE_KEY) : null)
          : await SecureStore.getItemAsync(STORAGE_KEY);
        if (!seen && !cancelled) setVisible(true);
      } catch {
        // If storage blows up, fail closed (don't show — better than spamming).
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Animate step transition.
  useEffect(() => {
    if (!visible) return;
    opacity.setValue(0);
    slide.setValue(16);
    Animated.parallel([
      Animated.timing(opacity, {
        toValue: 1, duration: 280, easing: Easing.out(Easing.ease), useNativeDriver: true,
      }),
      Animated.timing(slide, {
        toValue: 0, duration: 320, easing: Easing.out(Easing.ease), useNativeDriver: true,
      }),
    ]).start();
  }, [step, visible, opacity, slide]);

  const finish = async () => {
    if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    setVisible(false);
    try {
      if (Platform.OS === 'web') {
        if (typeof window !== 'undefined') window.localStorage.setItem(STORAGE_KEY, '1');
      } else {
        await SecureStore.setItemAsync(STORAGE_KEY, '1');
      }
    } catch { /* persist best-effort */ }
  };

  const next = () => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    if (step < STEPS.length - 1) setStep(step + 1);
    else finish();
  };

  const prev = () => {
    if (step === 0) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setStep(step - 1);
  };

  if (!visible) return null;
  const s = STEPS[step];
  const isLast = step === STEPS.length - 1;
  const isFirst = step === 0;

  return (
    <Modal visible animationType="fade" transparent statusBarTranslucent onRequestClose={finish}>
      <View style={styles.scrim}>
        <Animated.View style={[styles.card, { opacity, transform: [{ translateY: slide }] }]}>
          <View style={styles.iconWrap}>
            <Ionicons name={s.icon} size={28} color={colors.accent} />
          </View>

          <Text style={styles.title}>{s.title}</Text>
          <Text style={styles.body}>{s.body}</Text>

          {/* Step dots */}
          <View style={styles.dotsRow}>
            {STEPS.map((_, i) => (
              <View
                key={i}
                style={[
                  styles.dot,
                  i === step && styles.dotActive,
                ]}
              />
            ))}
          </View>

          {/* Buttons */}
          <View style={styles.btnRow}>
            <Pressable
              onPress={finish}
              style={({ pressed }) => [styles.skipBtn, pressed && { opacity: 0.6 }]}
              hitSlop={8}
            >
              <Text style={styles.skipText}>Skip</Text>
            </Pressable>

            <View style={{ flexDirection: 'row', gap: 8 }}>
              {!isFirst && (
                <Pressable
                  onPress={prev}
                  style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.7 }]}
                  hitSlop={8}
                >
                  <Ionicons name="chevron-back" size={20} color={colors.text} />
                </Pressable>
              )}
              <Pressable
                onPress={next}
                style={({ pressed }) => [styles.nextBtn, pressed && { opacity: 0.85 }]}
              >
                <Text style={styles.nextText}>{isLast ? 'Get started' : 'Next'}</Text>
                {!isLast && <Ionicons name="chevron-forward" size={18} color="#0E0F11" />}
              </Pressable>
            </View>
          </View>
        </Animated.View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  scrim: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.78)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: spacing.xl,
  },
  card: {
    width: '100%',
    maxWidth: 380,
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.borderHi,
    padding: spacing.xl,
    ...shadow.lift,
  },
  iconWrap: {
    width: 56,
    height: 56,
    borderRadius: 16,
    backgroundColor: colors.cardHi,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  title: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 22,
    letterSpacing: -0.4,
    marginBottom: spacing.sm,
  },
  body: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    lineHeight: 21,
    marginBottom: spacing.xl,
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 6,
    marginBottom: spacing.xl,
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: colors.cardHi,
  },
  dotActive: {
    backgroundColor: colors.accent,
    width: 18,
  },
  btnRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  skipBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  skipText: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 14,
  },
  backBtn: {
    width: 44,
    height: 44,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
  },
  nextBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 18,
    height: 44,
    borderRadius: radius.md,
    backgroundColor: colors.text,
  },
  nextText: {
    color: '#0E0F11',
    fontFamily: font.semibold,
    fontSize: 15,
  },
});
