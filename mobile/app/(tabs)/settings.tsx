import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Linking, Platform } from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient, Stop, Rect } from 'react-native-svg';

import { Screen } from '@/components/Screen';
import { Pill } from '@/components/Pill';
import { PrimaryButton } from '@/components/PrimaryButton';
import { Logo } from '@/components/Logo';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';

export default function SettingsScreen() {
  const { user, signOut } = useAuth();
  const [copied, setCopied] = useState(false);
  const profBgId = `prof-bg-${React.useId().replace(/:/g, '')}`;

  const onCopyUid = useCallback(async () => {
    if (!user?.uid) return;
    try {
      await Clipboard.setStringAsync(user.uid);
      setCopied(true);
      if (Platform.OS !== 'web') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      }
      setTimeout(() => setCopied(false), 1600);
    } catch {}
  }, [user?.uid]);

  const onSignOut = useCallback(() => {
    Alert.alert(
      'Sign out?',
      'You will need to enter your credentials again to sign back in.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign out', style: 'destructive', onPress: () => { signOut(); } },
      ],
    );
  }, [signOut]);

  const isPro = !!user?.is_pro;

  return (
    <Screen title="Account" subtitle="Your TradeHub profile and app settings.">
      {/* Profile card with gradient + glow */}
      <View style={[styles.profileCard, glow.card]}>
        <Svg style={StyleSheet.absoluteFill}>
          <Defs>
            <LinearGradient id={profBgId} x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0" stopColor="#1a2238" />
              <Stop offset="0.55" stopColor="#141b2e" />
              <Stop offset="1" stopColor="#0f1524" />
            </LinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill={`url(#${profBgId})`} />
        </Svg>
        <View style={styles.profileInner}>
          <View style={styles.avatarRow}>
            <Logo size={56} />
            <View style={{ flex: 1, marginLeft: spacing.md }}>
              <Text style={styles.displayName} numberOfLines={1}>{user?.display || 'Trader'}</Text>
              <View style={styles.pillRow}>
                <Pill
                  label={isPro ? `${user.plan.toUpperCase()} PLAN` : 'FREE PLAN'}
                  tone={isPro ? 'accent' : 'neutral'}
                  small
                />
                {user?.is_admin ? <Pill label="Admin" tone="warning" small /> : null}
              </View>
            </View>
          </View>

          <Pressable onPress={onCopyUid} style={({ pressed }) => [styles.uidBox, pressed && { opacity: 0.8 }]}>
            <View style={{ flex: 1 }}>
              <Text style={styles.uidLabel}>YOUR UID</Text>
              <Text style={styles.uidValue}>{user?.uid || '—'}</Text>
            </View>
            <View style={[styles.copyChip, copied && styles.copyChipActive]}>
              <Ionicons
                name={copied ? 'checkmark' : 'copy-outline'}
                size={14}
                color={copied ? colors.positive : colors.textDim}
              />
              <Text style={[styles.copyChipText, copied && { color: colors.positive }]}>
                {copied ? 'Copied' : 'Copy'}
              </Text>
            </View>
          </Pressable>
        </View>
      </View>

      {/* Quick links */}
      <Text style={styles.sectionLabel}>Open in browser</Text>
      <View style={styles.section}>
        <SettingsLink
          icon="globe-outline"
          label="Web portal"
          hint="Build strategies, scan, manage subscriptions"
          onPress={() => Linking.openURL('https://tradehub.markets').catch(() => {})}
        />
        <SettingsLink
          icon="paper-plane-outline"
          label="Telegram bot"
          hint="Live trade alerts in your DMs"
          onPress={() => Linking.openURL('https://t.me/tradehub_bot').catch(() => {})}
        />
        <SettingsLink
          icon="help-circle-outline"
          label="Support & feedback"
          hint="Get help or send a feature request"
          onPress={() => Linking.openURL('mailto:hi@tradehub.markets').catch(() => {})}
        />
      </View>

      <View style={[styles.section, { marginTop: spacing.xl }]}>
        <PrimaryButton label="Sign out" variant="destructive" onPress={onSignOut} />
      </View>

      <Text style={styles.version}>TradeHub Mobile · v1.1.0 · early access</Text>
    </Screen>
  );
}

function SettingsLink({
  icon, label, hint, onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  hint?: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.linkRow, pressed && { opacity: 0.85, transform: [{ scale: 0.997 }] }]}
    >
      <View style={styles.linkIcon}>
        <Ionicons name={icon} size={20} color={colors.accent} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.linkLabel}>{label}</Text>
        {hint ? <Text style={styles.linkHint}>{hint}</Text> : null}
      </View>
      <Ionicons name="chevron-forward" size={18} color={colors.textMute} />
    </Pressable>
  );
}

const styles = StyleSheet.create({
  profileCard: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  profileInner: { padding: spacing.lg },
  avatarRow: { flexDirection: 'row', alignItems: 'center' },
  displayName: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 19,
    letterSpacing: -0.3,
  },
  pillRow: { flexDirection: 'row', gap: 6, marginTop: 8 },
  uidBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: 14,
    paddingVertical: 12,
    gap: spacing.md,
  },
  uidLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.7,
    marginBottom: 4,
  },
  uidValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 15,
    letterSpacing: 0.6,
  },
  copyChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.cardHi,
  },
  copyChipActive: {
    borderColor: 'rgba(52,211,153,0.5)',
    backgroundColor: colors.positiveDim,
  },
  copyChipText: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.4,
  },
  sectionLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginTop: spacing.xl,
    marginBottom: spacing.sm,
  },
  section: { gap: spacing.sm },
  linkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.md,
  },
  linkIcon: {
    width: 40,
    height: 40,
    borderRadius: radius.md,
    backgroundColor: colors.accentDim,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.28)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  linkLabel: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 15,
  },
  linkHint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    marginTop: 2,
  },
  version: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    textAlign: 'center',
    marginTop: spacing.xxl,
  },
});
