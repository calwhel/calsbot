import React, { useCallback, useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Linking, Platform, ActivityIndicator, Switch, TextInput } from 'react-native';
import { useFocusEffect, useRouter } from 'expo-router';
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
import { RiskDisclaimer } from '@/components/RiskDisclaimer';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPut, apiDelete, apiPostFlex, type Portfolio, type PushPrefs } from '@/lib/api';

export default function SettingsScreen() {
  const { user, uid, signOut, refreshUser } = useAuth();
  const router = useRouter();
  const [copied, setCopied] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshNote, setRefreshNote] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Re-validate the user payload every time the Account tab regains focus,
  // so a Pro upgrade made on the web shows up immediately when the user
  // jumps back to this screen. Throttled by AuthContext (15s window).
  const lastFocusRefRef = useRef<number>(0);
  useFocusEffect(
    useCallback(() => {
      if (!uid) return;
      const now = Date.now();
      if (now - lastFocusRefRef.current < 10_000) return;
      lastFocusRefRef.current = now;
      refreshUser().catch(() => {});
    }, [uid, refreshUser]),
  );

  const onRefreshStatus = useCallback(async () => {
    if (refreshing) return;
    setRefreshing(true);
    setRefreshNote(null);
    try {
      const before = !!user?.is_pro;
      const fresh = await refreshUser();
      if (Platform.OS !== 'web') {
        Haptics.selectionAsync().catch(() => {});
      }
      if (fresh) {
        if (!!fresh.is_pro !== before) {
          setRefreshNote(fresh.is_pro ? 'Pro plan detected — unlocked!' : 'Plan updated.');
        } else {
          setRefreshNote('Account is up to date.');
        }
      } else {
        setRefreshNote("Couldn't reach the server.");
      }
    } finally {
      setRefreshing(false);
      setTimeout(() => setRefreshNote(null), 3500);
    }
  }, [refreshing, user?.is_pro, refreshUser]);
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

  const planLabel = 'TRADEHUB';

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
              <Stop offset="0" stopColor={colors.card} />
              <Stop offset="0.5" stopColor={colors.card} />
              <Stop offset="1" stopColor={colors.card} />
            </SvgLinearGradient>
            <RadialGradient id={orbId} cx="100%" cy="0%" rx="65%" ry="65%">
              <Stop offset="0" stopColor={colors.card} stopOpacity="0" />
              <Stop offset="0.55" stopColor={colors.card} stopOpacity="0" />
              <Stop offset="1" stopColor={colors.card} stopOpacity="0" />
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
                  tone={'neutral'}
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

          {/* Refresh-status row — lets a user who just upgraded on the web
              pull a fresh entitlement payload without signing out. */}
          <Pressable
            onPress={onRefreshStatus}
            disabled={refreshing}
            style={({ pressed }) => [
              styles.refreshRow,
              pressed && !refreshing && { opacity: 0.85 },
            ]}
          >
            {refreshing ? (
              <ActivityIndicator size="small" color={colors.accent} />
            ) : (
              <Ionicons name="refresh" size={14} color={colors.accent} />
            )}
            <Text style={styles.refreshLabel}>
              {refreshing ? 'Checking subscription…' : (refreshNote || 'Refresh subscription status')}
            </Text>
          </Pressable>

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


      {/* Trading account — balance + lot size for pip/$ display */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Trading account" />
      </View>
      <TradingAccountCard uid={uid} />

      {/* Push notification preferences */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Push notifications" />
      </View>
      <PushPrefsCard uid={uid} />

      {/* Strategy maintenance — no code pasting required */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Strategy tools" />
      </View>
      <StrategyToolsCard uid={uid} />

      {/* Broker connections */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Brokers" />
      </View>
      <View style={styles.section}>
        <SettingsLink
          icon="trending-up-outline"
          tone="accent"
          label="Bitunix — Crypto"
          hint="Connect your Bitunix account to trade crypto strategies live"
          onPress={() => router.push('/bitunix' as any)}
        />
        <SettingsLink
          icon="globe-outline"
          tone="accent"
          label="cTrader — FP Markets (Forex & Indices)"
          hint="Connect your FP Markets cTrader account to trade forex and index strategies live"
          onPress={() => router.push('/ctrader' as any)}
        />
      </View>

      {/* Earn — affiliate program */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Earn" />
      </View>
      <View style={styles.section}>
        <SettingsLink
          icon="diamond-outline"
          tone="warning"
          label="Affiliate program"
          hint="Earn 30% of subs + 20% of fees from every referral"
          onPress={() => router.push('/affiliate' as any)}
        />
      </View>

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

      {/* Legal */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Legal" />
      </View>
      <View style={styles.section}>
        <SettingsLink
          icon="document-text-outline"
          tone="accent"
          label="Privacy Policy"
          onPress={() => Linking.openURL('https://tradehubmarkets.com/privacy').catch(() => {})}
        />
        <SettingsLink
          icon="shield-checkmark-outline"
          tone="accent"
          label="Terms & Conditions"
          onPress={() => Linking.openURL('https://tradehubmarkets.com/terms').catch(() => {})}
        />
      </View>

      <RiskDisclaimer />

      <View style={[styles.section, { marginTop: spacing.xl }]}>
        <PrimaryButton label="Sign out" variant="destructive" onPress={onSignOut} />
      </View>

      <View style={[styles.section, { marginTop: spacing.md }]}>
        <Pressable
          onPress={() => {
            Alert.alert(
              'Delete Account',
              'This will permanently delete your account, deactivate all strategies, and remove your personal data. This action cannot be undone.',
              [
                { text: 'Cancel', style: 'cancel' },
                {
                  text: 'Delete my account',
                  style: 'destructive',
                  onPress: async () => {
                    if (!uid) return;
                    setDeleting(true);
                    try {
                      await apiDelete('/api/mobile/account', uid);
                      Alert.alert('Account deleted', 'Your account has been deleted.');
                      signOut();
                    } catch (e: any) {
                      Alert.alert('Could not delete', e?.message || 'Please try again or contact support.');
                    } finally {
                      setDeleting(false);
                    }
                  },
                },
              ],
            );
          }}
          disabled={deleting}
          style={({ pressed }) => [styles.deleteBtn, pressed && { opacity: 0.85 }]}
        >
          {deleting
            ? <ActivityIndicator size="small" color={colors.negative} />
            : <Ionicons name="trash-outline" size={16} color={colors.negative} />}
          <Text style={styles.deleteText}>{deleting ? 'Deleting...' : 'Delete account'}</Text>
        </Pressable>
      </View>

      <Text style={styles.version}>TradeHub Mobile · v1.1.2 · early access</Text>
    </Screen>
  );
}

const MIN_USD_OPTIONS = [0, 10, 25, 50, 100] as const;

type RaiseCapResponse = {
  ok?: boolean;
  raised?: number;
  already_at_or_above?: number;
  max_trades_per_day?: number;
  detail?: string;
  error?: string;
};

type HealResponse = {
  ok?: boolean;
  promoted_to_paper?: number;
  stale_expired?: number;
  orphan_forex_expired?: number;
  detail?: string;
  error?: string;
};

function StrategyToolsCard({ uid }: { uid: string | null | undefined }) {
  const [busy, setBusy] = useState<'cap' | 'heal' | null>(null);

  const onRaiseDailyCap = useCallback(() => {
    if (!uid || busy) return;
    Alert.alert(
      'Raise daily trade limit?',
      'Sets max trades per day to 10 on every strategy currently below 10. Takes effect immediately — strategies that already hit today\'s low cap can fire again.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Raise to 10/day',
          onPress: async () => {
            setBusy('cap');
            try {
              const r = await apiPostFlex<RaiseCapResponse>(
                '/api/strategies/raise-daily-cap',
                { uid, max_trades_per_day: 10 },
              );
              if (r.ok) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                Alert.alert(
                  'Daily cap raised',
                  `Updated ${r.raised ?? 0} strategies to ${r.max_trades_per_day ?? 10} trades/day.` +
                    ((r.already_at_or_above ?? 0) > 0
                      ? ` ${r.already_at_or_above} were already at or above that limit.`
                      : ''),
                );
              } else {
                Alert.alert('Could not update', r.detail || r.error || 'The server may need an update first — try the web portal.');
              }
            } catch (e: unknown) {
              const msg = e instanceof Error ? e.message : 'Request failed';
              Alert.alert('Error', msg);
            } finally {
              setBusy(null);
            }
          },
        },
      ],
    );
  }, [uid, busy]);

  const onRepairStrategies = useCallback(() => {
    if (!uid || busy) return;
    Alert.alert(
      'Repair strategies?',
      'Promotes draft strategies to paper, fixes missing symbols, and clears stuck open trades that block new entries.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Repair now',
          onPress: async () => {
            setBusy('heal');
            try {
              const r = await apiPostFlex<HealResponse>(
                '/api/strategies/ensure-firing',
                { uid },
              );
              if (r.ok) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                const parts = [
                  r.promoted_to_paper ? `${r.promoted_to_paper} promoted to paper` : null,
                  r.stale_expired ? `${r.stale_expired} stale opens cleared` : null,
                  r.orphan_forex_expired ? `${r.orphan_forex_expired} orphan forex rows cleared` : null,
                ].filter(Boolean);
                Alert.alert(
                  'Repair complete',
                  parts.length ? parts.join(' · ') : 'No changes needed — strategies look healthy.',
                );
              } else {
                Alert.alert('Repair failed', r.detail || r.error || 'Try again later.');
              }
            } catch (e: unknown) {
              const msg = e instanceof Error ? e.message : 'Request failed';
              Alert.alert('Error', msg);
            } finally {
              setBusy(null);
            }
          },
        },
      ],
    );
  }, [uid, busy]);

  return (
    <View style={styles.section}>
      <SettingsLink
        icon="trending-up-outline"
        tone="positive"
        label={busy === 'cap' ? 'Raising daily cap…' : 'Raise daily trade limit'}
        hint="Set all strategies to 10 trades/day (if currently lower)"
        onPress={onRaiseDailyCap}
      />
      <SettingsLink
        icon="build-outline"
        tone="warning"
        label={busy === 'heal' ? 'Repairing…' : 'Repair strategies'}
        hint="Clear stuck opens and fix strategies not scanning"
        onPress={onRepairStrategies}
      />
    </View>
  );
}

function PushPrefsCard({ uid }: { uid: string | null | undefined }) {
  const [paper, setPaper] = useState(true);
  const [live, setLive]   = useState(true);
  const [minUsd, setMinUsd] = useState<number>(0);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!uid) return;
    let cancel = false;
    (async () => {
      try {
        const s = await apiGet<PushPrefs & Record<string, unknown>>('/api/settings', uid);
        if (cancel) return;
        setPaper(s.push_notify_paper !== false);
        setLive(s.push_notify_live !== false);
        setMinUsd(Number(s.push_min_position_usd) || 0);
      } catch {
        // fail-silent — user can still toggle, save will surface errors
      } finally {
        if (!cancel) setLoaded(true);
      }
    })();
    return () => { cancel = true; };
  }, [uid]);

  // Optimistic-update with rollback: each handler captures the previous value
  // and reverts the local state if the PUT fails so the UI never sticks at an
  // unsaved value.
  const persist = useCallback(async (
    patch: Partial<PushPrefs>,
    rollback: () => void,
  ) => {
    if (!uid) { rollback(); return; }
    setSaving(true);
    try {
      await apiPut('/api/settings', patch, uid);
      if (Platform.OS !== 'web') {
        Haptics.selectionAsync().catch(() => {});
      }
    } catch (e: any) {
      rollback();
      Alert.alert('Could not save', e?.message || 'Please try again.');
    } finally {
      setSaving(false);
    }
  }, [uid]);

  const onTogglePaper = useCallback((v: boolean) => {
    const prev = paper;
    setPaper(v);
    persist({ push_notify_paper: v }, () => setPaper(prev));
  }, [paper, persist]);

  const onToggleLive = useCallback((v: boolean) => {
    const prev = live;
    setLive(v);
    persist({ push_notify_live: v }, () => setLive(prev));
  }, [live, persist]);

  const onPickMinUsd = useCallback((v: number) => {
    if (v === minUsd) return;
    const prev = minUsd;
    setMinUsd(v);
    persist({ push_min_position_usd: v }, () => setMinUsd(prev));
  }, [minUsd, persist]);

  return (
    <View style={pushStyles.card}>
      <View style={pushStyles.row}>
        <View style={[pushStyles.iconWrap, { backgroundColor: colors.violetDim, borderColor: 'rgba(255,255,255,0.10)' }]}>
          <Ionicons name="document-text-outline" size={18} color={colors.violet} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={pushStyles.label}>Paper trade alerts</Text>
          <Text style={pushStyles.hint}>Pushes when one of your strategies fires a paper trade.</Text>
        </View>
        <Switch
          value={paper}
          onValueChange={onTogglePaper}
          disabled={!loaded || saving}
          trackColor={{ true: colors.violet, false: '#22252A' }}
          thumbColor="#fff"
        />
      </View>

      <View style={pushStyles.divider} />

      <View style={pushStyles.row}>
        <View style={[pushStyles.iconWrap, { backgroundColor: colors.positiveDim, borderColor: 'rgba(52,211,153,0.32)' }]}>
          <Ionicons name="flash" size={18} color={colors.positive} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={pushStyles.label}>Live trade alerts</Text>
          <Text style={pushStyles.hint}>Pushes when a live order is placed (manual or strategy).</Text>
        </View>
        <Switch
          value={live}
          onValueChange={onToggleLive}
          disabled={!loaded || saving}
          trackColor={{ true: colors.positive, false: '#22252A' }}
          thumbColor="#fff"
        />
      </View>

      <View style={pushStyles.divider} />

      <View style={{ paddingHorizontal: 4 }}>
        <Text style={pushStyles.label}>Min position size</Text>
        <Text style={pushStyles.hint}>Skip pushes for trades below this notional. Set to $0 to get every alert.</Text>
        <View style={pushStyles.chipRow}>
          {MIN_USD_OPTIONS.map((v) => {
            const active = v === minUsd;
            return (
              <Pressable
                key={v}
                onPress={() => onPickMinUsd(v)}
                disabled={!loaded || saving}
                style={({ pressed }) => [
                  pushStyles.chip,
                  active && pushStyles.chipActive,
                  pressed && { opacity: 0.85 },
                ]}
              >
                <Text style={[pushStyles.chipText, active && pushStyles.chipTextActive]}>
                  {v === 0 ? 'All' : `$${v}+`}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>
    </View>
  );
}

function TradingAccountCard({ uid }: { uid: string | null | undefined }) {
  const [balance, setBalance] = useState('10000');
  const [lotSize, setLotSize] = useState(0.1);
  const [loaded, setLoaded] = useState(false);
  const [saving, setSaving] = useState(false);

  const LOT_OPTIONS = [
    { label: '0.01 micro', value: 0.01 },
    { label: '0.05',       value: 0.05 },
    { label: '0.1 mini',   value: 0.1  },
    { label: '0.5',        value: 0.5  },
    { label: '1.0 std',    value: 1.0  },
  ];

  useEffect(() => {
    if (!uid) return;
    let cancel = false;
    (async () => {
      try {
        const s = await apiGet<Record<string, any>>('/api/settings', uid);
        if (cancel) return;
        const bal = Number(s.account_balance);
        if (bal > 0) setBalance(bal.toFixed(0));
        const ls = Number(s.lot_size);
        if (ls > 0) setLotSize(ls);
      } catch { }
      finally { if (!cancel) setLoaded(true); }
    })();
    return () => { cancel = true; };
  }, [uid]);

  const persist = useCallback(async (patch: Record<string, any>) => {
    if (!uid) return;
    setSaving(true);
    try {
      await apiPut('/api/settings', patch, uid);
      if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    } catch (e: any) {
      Alert.alert('Could not save', e?.message || 'Please try again.');
    } finally {
      setSaving(false);
    }
  }, [uid]);

  const onBalanceBlur = useCallback(() => {
    const parsed = parseFloat(balance.replace(/,/g, ''));
    if (!isNaN(parsed) && parsed > 0) {
      setBalance(parsed.toFixed(0));
      persist({ account_balance: parsed });
    }
  }, [balance, persist]);

  const onLotPress = useCallback((v: number) => {
    setLotSize(v);
    persist({ lot_size: v });
  }, [persist]);

  if (!loaded) {
    return (
      <View style={taStyles.card}>
        <ActivityIndicator size="small" color={colors.accent} style={{ margin: spacing.lg }} />
      </View>
    );
  }

  return (
    <View style={taStyles.card}>
      <View style={taStyles.header}>
        <View style={[taStyles.iconWrap, { backgroundColor: colors.accentDim, borderColor: 'rgba(255,255,255,0.10)' }]}>
          <Ionicons name="wallet-outline" size={18} color={colors.accent} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={taStyles.title}>Trading account</Text>
          <Text style={taStyles.hint}>Sets the $ value shown alongside % P&L on strategy stats.</Text>
        </View>
        {saving ? <ActivityIndicator size="small" color={colors.accent} /> : null}
      </View>

      <View style={taStyles.divider} />

      {/* Account balance */}
      <View style={taStyles.fieldRow}>
        <Text style={taStyles.fieldLabel}>Account balance</Text>
        <View style={taStyles.inputWrap}>
          <Text style={taStyles.currencySymbol}>$</Text>
          <TextInput
            style={taStyles.input}
            value={balance}
            onChangeText={setBalance}
            onBlur={onBalanceBlur}
            keyboardType="numeric"
            returnKeyType="done"
            placeholderTextColor={colors.textMute}
          />
        </View>
      </View>

      <View style={taStyles.divider} />

      {/* Lot size */}
      <View style={{ paddingHorizontal: spacing.md, paddingBottom: spacing.md }}>
        <Text style={[taStyles.fieldLabel, { marginBottom: spacing.sm }]}>Default lot size</Text>
        <View style={taStyles.chipRow}>
          {LOT_OPTIONS.map((opt) => {
            const active = Math.abs(opt.value - lotSize) < 0.001;
            return (
              <Pressable
                key={opt.value}
                onPress={() => onLotPress(opt.value)}
                style={[taStyles.chip, active && taStyles.chipActive]}
              >
                <Text style={[taStyles.chipText, active && taStyles.chipTextActive]}>
                  {opt.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>
    </View>
  );
}

const taStyles = StyleSheet.create({
  card: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    marginTop: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing.md,
    padding: spacing.md,
  },
  iconWrap: {
    width: 36,
    height: 36,
    borderRadius: 10,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: { color: colors.text, fontFamily: font.bold, fontSize: 14 },
  hint:  { color: colors.textDim, fontFamily: font.regular, fontSize: 12, marginTop: 2, lineHeight: 17 },
  divider: { height: 1, backgroundColor: colors.border },
  fieldRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
  },
  fieldLabel: { color: colors.textDim, fontFamily: font.bold, fontSize: 13 },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.cardHi,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.sm,
    paddingVertical: 6,
    minWidth: 110,
  },
  currencySymbol: { color: colors.textDim, fontFamily: font.bold, fontSize: 14, marginRight: 4 },
  input: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
    flex: 1,
    minWidth: 80,
    padding: 0,
  },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: 'rgba(0,0,0,0.28)',
  },
  chipActive: {
    borderColor: 'rgba(63,182,139,0.50)',
    backgroundColor: colors.accentDim,
  },
  chipText: { color: colors.textDim, fontFamily: font.bold, fontSize: 12, letterSpacing: 0.2 },
  chipTextActive: { color: colors.accent },
});

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
    accent:   { color: colors.accent,   bg: colors.accentDim,   border: 'rgba(255,255,255,0.10)' },
    violet:   { color: colors.violet,   bg: colors.violetDim,   border: 'rgba(255,255,255,0.10)' },
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
    borderColor: 'rgba(255,255,255,0.10)',
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
    color: '#9A9BA0',
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

  refreshRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: spacing.lg,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: radius.md,
    backgroundColor: 'rgba(255,255,255,0.10)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
  },
  refreshLabel: {
    color: colors.accent,
    fontFamily: font.semibold,
    fontSize: 12,
    letterSpacing: 0.3,
    flex: 1,
  },
  uidBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.42)',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    borderRadius: radius.md,
    paddingHorizontal: 14,
    paddingVertical: 12,
    gap: spacing.md,
  },
  uidLabel: {
    color: '#9A9BA0',
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
    borderColor: 'rgba(255,255,255,0.10)',
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
    borderColor: colors.border,
    overflow: 'hidden',
    backgroundColor: colors.cardHi,
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
  deleteBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(229,72,77,0.24)',
    backgroundColor: 'rgba(229,72,77,0.06)',
  },
  deleteText: {
    color: colors.negative,
    fontFamily: font.semibold,
    fontSize: 14,
  },
});

const pushStyles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.md,
    ...glow.card,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    paddingVertical: 4,
  },
  iconWrap: {
    width: 36,
    height: 36,
    borderRadius: radius.md,
    borderWidth: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    letterSpacing: -0.2,
  },
  hint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11.5,
    marginTop: 2,
    lineHeight: 15,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(255,255,255,0.06)',
    marginVertical: spacing.md,
  },
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: spacing.md,
  },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: 'rgba(0,0,0,0.32)',
  },
  chipActive: {
    borderColor: 'rgba(255,255,255,0.10)',
    backgroundColor: colors.accentDim,
  },
  chipText: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.3,
  },
  chipTextActive: {
    color: colors.accent,
  },
});
