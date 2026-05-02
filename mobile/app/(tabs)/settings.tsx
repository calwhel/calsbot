import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Linking, Platform } from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, {
  Defs,
  LinearGradient as SvgLinearGradient,
  RadialGradient,
  Stop,
  Rect,
  Circle,
} from 'react-native-svg';
import { useQuery } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { Pill } from '@/components/Pill';
import { PrimaryButton } from '@/components/PrimaryButton';
import { Logo } from '@/components/Logo';
import { SectionLabel } from '@/components/SectionLabel';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type Portfolio } from '@/lib/api';

export default function SettingsScreen() {
  const { user, uid, signOut } = useAuth();
  const [copied, setCopied] = useState(false);
  const heroId = `prof-bg-${React.useId().replace(/:/g, '')}`;
  const orbId = `prof-orb-${heroId}`;

  // Fetch portfolio for the inline mini-stats on the profile card.
  const { data: portfolio } = useQuery({
    queryKey: ['portfolio', uid],
    queryFn: () => apiGet<Portfolio>('/api/portfolio', uid),
    enabled: !!uid,
  });

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
  const planLabel = isPro ? `${user.plan.toUpperCase()} PLAN` : 'FREE PLAN';

  return (
    <Screen
      title="Account"
      subtitle="Your TradeHub profile and app settings."
      ambient="violet"
    >
      {/* Premium profile hero with mesh gradient + orbs */}
      <View style={[styles.profileCard, glow.accent]}>
        <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
          <Defs>
            <SvgLinearGradient id={heroId} x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0" stopColor="#221c4a" />
              <Stop offset="0.5" stopColor="#161836" />
              <Stop offset="1" stopColor="#0a0e22" />
            </SvgLinearGradient>
            <RadialGradient id={orbId} cx="100%" cy="0%" rx="65%" ry="65%">
              <Stop offset="0" stopColor="#a78bfa" stopOpacity="0.42" />
              <Stop offset="0.55" stopColor="#22d3ee" stopOpacity="0.18" />
              <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
            </RadialGradient>
          </Defs>
          <Rect width="100" height="100" fill={`url(#${heroId})`} />
          <Rect width="100" height="100" fill={`url(#${orbId})`} />
        </Svg>

        {/* Decorative orbital rings (top-right) */}
        <View pointerEvents="none" style={styles.ringsWrap}>
          <Svg width={150} height={150}>
            <Circle cx={75} cy={75} r={60} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
            <Circle cx={75} cy={75} r={42} fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth={1} />
            <Circle cx={75} cy={75} r={26} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth={1} />
          </Svg>
        </View>

        <View style={styles.profileInner}>
          <View style={styles.avatarRow}>
            <View style={styles.logoChip}>
              <Logo size={48} />
            </View>
            <View style={{ flex: 1, marginLeft: spacing.md }}>
              <Text style={styles.displayName} numberOfLines={1}>
                {user?.display || 'Trader'}
              </Text>
              <View style={styles.pillRow}>
                <Pill
                  label={planLabel}
                  tone={isPro ? 'accent' : 'neutral'}
                  small
                />
                {user?.is_admin ? <Pill label="Admin" tone="warning" small /> : null}
              </View>
            </View>
          </View>

          {/* Mini-stats — only render when we have data */}
          {portfolio ? (
            <View style={styles.statsStrip}>
              <View style={styles.statCell}>
                <Text style={styles.statValue}>{portfolio.total_strategies}</Text>
                <Text style={styles.statLabel}>STRATEGIES</Text>
              </View>
              <View style={styles.statSep} />
              <View style={styles.statCell}>
                <Text style={styles.statValue}>{portfolio.total_trades}</Text>
                <Text style={styles.statLabel}>TRADES</Text>
              </View>
              <View style={styles.statSep} />
              <View style={styles.statCell}>
                <Text
                  style={[
                    styles.statValue,
                    {
                      color:
                        portfolio.pnl_all > 0.01
                          ? colors.positive
                          : portfolio.pnl_all < -0.01
                            ? colors.negative
                            : colors.text,
                    },
                  ]}
                >
                  {portfolio.pnl_all > 0 ? '+' : ''}{portfolio.pnl_all.toFixed(1)}%
                </Text>
                <Text style={styles.statLabel}>ALL-TIME</Text>
              </View>
            </View>
          ) : null}

          <Pressable onPress={onCopyUid} style={({ pressed }) => [styles.uidBox, pressed && { opacity: 0.85 }]}>
            <View style={{ flex: 1 }}>
              <Text style={styles.uidLabel}>YOUR UID · API KEY</Text>
              <Text style={styles.uidValue}>{user?.uid || '—'}</Text>
            </View>
            <View style={[styles.copyChip, copied && styles.copyChipActive]}>
              <Ionicons
                name={copied ? 'checkmark' : 'copy-outline'}
                size={14}
                color={copied ? colors.positive : colors.accent}
              />
              <Text style={[styles.copyChipText, copied && { color: colors.positive }]}>
                {copied ? 'Copied' : 'Copy'}
              </Text>
            </View>
          </Pressable>
        </View>
      </View>

      {/* Pro upgrade card — only for free users */}
      {!isPro ? (
        <Pressable
          onPress={() => Linking.openURL('https://tradehub.markets/pricing').catch(() => {})}
          style={({ pressed }) => [
            styles.proCard,
            glow.accent,
            pressed && { opacity: 0.9, transform: [{ scale: 0.99 }] },
          ]}
        >
          <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
            <Defs>
              <SvgLinearGradient id="pro-bg" x1="0" y1="0" x2="1" y2="1">
                <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.22" />
                <Stop offset="0.6" stopColor="#7c3aed" stopOpacity="0.18" />
                <Stop offset="1" stopColor="#0a1024" stopOpacity="0.6" />
              </SvgLinearGradient>
            </Defs>
            <Rect width="100" height="100" fill="#0f1428" />
            <Rect width="100" height="100" fill="url(#pro-bg)" />
          </Svg>
          <View style={styles.proInner}>
            <View style={styles.proIcon}>
              <Ionicons name="diamond" size={20} color={colors.accentText} />
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.proTitle}>Upgrade to TradeHub Pro</Text>
              <Text style={styles.proHint}>Unlimited backtests, live alerts, and the full scanner.</Text>
            </View>
            <Ionicons name="arrow-forward-circle" size={26} color={colors.accent} />
          </View>
        </Pressable>
      ) : null}

      {/* Quick links */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Open in browser" />
      </View>
      <View style={styles.section}>
        <SettingsLink
          icon="globe-outline"
          tone="accent"
          label="Web portal"
          hint="Build strategies, scan, manage subscriptions"
          onPress={() => Linking.openURL('https://tradehub.markets').catch(() => {})}
        />
        <SettingsLink
          icon="paper-plane-outline"
          tone="violet"
          label="Telegram bot"
          hint="Live trade alerts in your DMs"
          onPress={() => Linking.openURL('https://t.me/tradehub_bot').catch(() => {})}
        />
        <SettingsLink
          icon="help-circle-outline"
          tone="positive"
          label="Support & feedback"
          hint="Get help or send a feature request"
          onPress={() => Linking.openURL('mailto:hi@tradehub.markets').catch(() => {})}
        />
      </View>

      <View style={[styles.section, { marginTop: spacing.xl }]}>
        <PrimaryButton label="Sign out" variant="destructive" onPress={onSignOut} />
      </View>

      <Text style={styles.version}>TradeHub Mobile · v1.1.2 · early access</Text>
    </Screen>
  );
}

function SettingsLink({
  icon, label, hint, tone = 'accent', onPress,
}: {
  icon: keyof typeof Ionicons.glyphMap;
  label: string;
  hint?: string;
  tone?: 'accent' | 'violet' | 'positive' | 'warning';
  onPress: () => void;
}) {
  const palette = {
    accent:   { color: colors.accent,   bg: colors.accentDim,   border: 'rgba(34,211,238,0.32)' },
    violet:   { color: colors.violet,   bg: colors.violetDim,   border: 'rgba(167,139,250,0.32)' },
    positive: { color: colors.positive, bg: colors.positiveDim, border: 'rgba(52,211,153,0.32)' },
    warning:  { color: colors.warning,  bg: colors.warningDim,  border: 'rgba(251,191,36,0.32)' },
  }[tone];
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.linkRow, pressed && { opacity: 0.85, transform: [{ scale: 0.997 }] }]}
    >
      <View
        style={[
          styles.linkIcon,
          { backgroundColor: palette.bg, borderColor: palette.border },
        ]}
      >
        <Ionicons name={icon} size={20} color={palette.color} />
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
    borderColor: 'rgba(167,139,250,0.28)',
    overflow: 'hidden',
    backgroundColor: colors.card,
    minHeight: 220,
  },
  ringsWrap: {
    position: 'absolute',
    top: -22,
    right: -22,
    opacity: 0.7,
  },
  profileInner: { padding: spacing.lg },
  avatarRow: { flexDirection: 'row', alignItems: 'center' },
  logoChip: {
    width: 60,
    height: 60,
    borderRadius: 18,
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  displayName: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 22,
    letterSpacing: -0.5,
  },
  pillRow: { flexDirection: 'row', gap: 6, marginTop: 8, flexWrap: 'wrap' },

  statsStrip: {
    flexDirection: 'row',
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
    borderRadius: radius.lg,
    paddingVertical: spacing.md,
    marginTop: spacing.lg,
  },
  statCell: { flex: 1, alignItems: 'center' },
  statValue: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 19,
    letterSpacing: -0.4,
    fontVariant: ['tabular-nums'],
  },
  statLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.7,
    marginTop: 3,
  },
  statSep: {
    width: 1,
    backgroundColor: 'rgba(255,255,255,0.08)',
    marginVertical: 6,
  },

  uidBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.42)',
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.18)',
    borderRadius: radius.md,
    paddingHorizontal: 14,
    paddingVertical: 12,
    gap: spacing.md,
  },
  uidLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.8,
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
    borderColor: 'rgba(34,211,238,0.32)',
    backgroundColor: colors.accentDim,
  },
  copyChipActive: {
    borderColor: 'rgba(52,211,153,0.5)',
    backgroundColor: colors.positiveDim,
  },
  copyChipText: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.4,
  },

  proCard: {
    marginTop: spacing.lg,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.32)',
    overflow: 'hidden',
    backgroundColor: '#0f1428',
  },
  proInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.lg,
  },
  proIcon: {
    width: 40,
    height: 40,
    borderRadius: 12,
    backgroundColor: colors.accent,
    alignItems: 'center',
    justifyContent: 'center',
  },
  proTitle: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 15,
    letterSpacing: -0.3,
  },
  proHint: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 12.5,
    marginTop: 2,
    lineHeight: 17,
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
    ...glow.card,
  },
  linkIcon: {
    width: 42,
    height: 42,
    borderRadius: radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  linkLabel: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 15,
    letterSpacing: -0.2,
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
