import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Pressable, Alert, Linking, Platform } from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { Screen } from '@/components/Screen';
import { Pill } from '@/components/Pill';
import { PrimaryButton } from '@/components/PrimaryButton';
import { colors, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';

export default function SettingsScreen() {
  const { user, signOut } = useAuth();
  const [copied, setCopied] = useState(false);

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
      'You will need to enter your UID again to sign back in.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Sign out', style: 'destructive', onPress: () => { signOut(); } },
      ],
    );
  }, [signOut]);

  return (
    <Screen title="Account" subtitle="Your TradeHub profile and app settings.">
      {/* Profile card */}
      <View style={styles.card}>
        <View style={styles.avatarRow}>
          <View style={styles.avatar}>
            <Text style={styles.avatarText}>
              {(user?.display || user?.uid || '?').charAt(0).toUpperCase()}
            </Text>
          </View>
          <View style={{ flex: 1, marginLeft: spacing.md }}>
            <Text style={styles.displayName} numberOfLines={1}>{user?.display || 'Trader'}</Text>
            <View style={styles.pillRow}>
              <Pill
                label={user?.is_pro ? `${user.plan.toUpperCase()} PLAN` : 'FREE PLAN'}
                tone={user?.is_pro ? 'accent' : 'neutral'}
                small
              />
              {user?.is_admin ? <Pill label="Admin" tone="warning" small /> : null}
            </View>
          </View>
        </View>

        <Pressable onPress={onCopyUid} style={({ pressed }) => [styles.uidBox, pressed && { opacity: 0.7 }]}>
          <View style={{ flex: 1 }}>
            <Text style={styles.uidLabel}>YOUR UID</Text>
            <Text style={styles.uidValue}>{user?.uid || '—'}</Text>
          </View>
          <Ionicons name={copied ? 'checkmark' : 'copy-outline'} size={18} color={copied ? colors.positive : colors.textDim} />
        </Pressable>
      </View>

      {/* Links */}
      <View style={styles.section}>
        <SettingsLink
          icon="globe-outline"
          label="Open web portal"
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

      <View style={styles.section}>
        <PrimaryButton label="Sign out" variant="destructive" onPress={onSignOut} />
      </View>

      <Text style={styles.version}>TradeHub Mobile · v1.0.0 · early access</Text>
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
    <Pressable onPress={onPress} style={({ pressed }) => [styles.linkRow, pressed && { opacity: 0.7 }]}>
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
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
  },
  avatarRow: { flexDirection: 'row', alignItems: 'center' },
  avatar: {
    width: 56, height: 56, borderRadius: 28,
    backgroundColor: colors.accentDim,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: { color: colors.accent, fontSize: 22, fontWeight: '800' },
  displayName: { color: colors.text, fontSize: 18, fontWeight: '700' },
  pillRow: { flexDirection: 'row', gap: 6, marginTop: 6 },
  uidBox: {
    flexDirection: 'row', alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: 14, paddingVertical: 12,
  },
  uidLabel: {
    color: colors.textMute, fontSize: 10, fontWeight: '700',
    letterSpacing: 0.6, marginBottom: 2,
  },
  uidValue: { color: colors.text, fontSize: 15, fontWeight: '700', letterSpacing: 0.6 },
  section: { marginTop: spacing.xl, gap: spacing.sm },
  linkRow: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.md,
    gap: spacing.md,
  },
  linkIcon: {
    width: 38, height: 38, borderRadius: radius.md,
    backgroundColor: colors.accentDim,
    alignItems: 'center', justifyContent: 'center',
  },
  linkLabel: { color: colors.text, fontSize: 15, fontWeight: '600' },
  linkHint: { color: colors.textMute, fontSize: 12, marginTop: 2 },
  version: {
    color: colors.textMute,
    fontSize: 11,
    textAlign: 'center',
    marginTop: spacing.xxl,
  },
});
