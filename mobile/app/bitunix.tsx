import React, { useState } from 'react';
import {
  View, Text, StyleSheet, Pressable, Alert, Linking,
  TextInput, ActivityIndicator, ScrollView,
} from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import * as Clipboard from 'expo-clipboard';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { PrimaryButton } from '@/components/PrimaryButton';
import { SectionLabel } from '@/components/SectionLabel';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPost, Portfolio, ApiError } from '@/lib/api';

const REFERRAL_URL = 'https://www.bitunix.com/register?vipCode=tradehubsave';
const FP_REFERRAL_URL = 'https://www.fpmarkets.com/?fpm-affiliate-utm-source=IB&fpm-affiliate-model=revenue-sharing';

type PortalSettings = {
  bitunix_uid?: string;
  bitunix_keys_set?: boolean;
};

export default function BitunixScreen() {
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();

  const [uidInput, setUidInput] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [apiSecret, setApiSecret] = useState('');
  const [showKeys, setShowKeys] = useState(false);

  const { data: portfolio, isLoading } = useQuery({
    queryKey: ['portfolio', uid],
    queryFn: () => apiGet<Portfolio>('/api/portfolio', uid),
    enabled: !!uid,
    staleTime: 30_000,
  });

  const { data: settings } = useQuery({
    queryKey: ['portal-settings', uid],
    queryFn: () => apiGet<PortalSettings>('/api/settings', uid),
    enabled: !!uid,
  });

  const aff = portfolio?.affiliate;
  const isAffiliated = aff?.ok ?? false;
  const hasUid = aff?.has_uid ?? false;
  const hasKeys = aff?.has_keys ?? false;
  const notAffiliated = hasUid && !isAffiliated && aff?.reason === 'uid_not_under_master';

  const saveMut = useMutation({
    mutationFn: async (body: Record<string, string>) => {
      await apiPost('/api/settings', body, uid);
    },
    onSuccess: async () => {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      qc.invalidateQueries({ queryKey: ['portfolio', uid] });
      qc.invalidateQueries({ queryKey: ['portal-settings', uid] });
      Alert.alert('Saved', 'Your Bitunix details have been updated.');
      setUidInput('');
      setApiKey('');
      setApiSecret('');
    },
    onError: (err: unknown) => {
      const msg = err instanceof ApiError ? err.message : 'Save failed. Please try again.';
      Alert.alert('Error', msg);
    },
  });

  const handleSave = () => {
    const body: Record<string, string> = {};
    if (uidInput.trim()) body.bitunix_uid = uidInput.trim();
    if (apiKey.trim()) body.bitunix_api_key = apiKey.trim();
    if (apiSecret.trim()) body.bitunix_api_secret = apiSecret.trim();
    if (!Object.keys(body).length) {
      Alert.alert('Nothing to save', 'Enter your UID or API keys first.');
      return;
    }
    saveMut.mutate(body);
  };

  const openReferral = () => Linking.openURL(REFERRAL_URL).catch(() => {});
  const openFPReferral = () => Linking.openURL(FP_REFERRAL_URL).catch(() => {});

  const copyReferral = async () => {
    await Clipboard.setStringAsync(REFERRAL_URL);
    await Haptics.selectionAsync();
    Alert.alert('Copied', 'Bitunix sign-up link copied to clipboard.');
  };

  return (
    <Screen title="Bitunix — Crypto" scroll>
      <Stack.Screen options={{ title: 'Bitunix — Crypto' }} />

      {/* Hero */}
      <View style={styles.hero}>
        <View style={[styles.heroIcon, isAffiliated && styles.heroIconOk]}>
          <Ionicons
            name={isAffiliated ? 'shield-checkmark' : 'trending-up-outline'}
            size={32}
            color={isAffiliated ? colors.positive : colors.text}
          />
        </View>
        <Text style={styles.heroTitle}>Bitunix · Crypto Live Trading</Text>
        <Text style={styles.heroSub}>
          Connect your Bitunix account to run crypto strategies with real execution.
          You must be registered under the TradeHub affiliate link to unlock live mode.
        </Text>
      </View>

      {isLoading ? (
        <ActivityIndicator color={colors.positive} style={{ marginTop: spacing.xl }} />
      ) : (
        <>
          {/* ── Status card ── */}
          {isAffiliated ? (
            <View style={[styles.card, styles.cardOk]}>
              <View style={styles.statusRow}>
                <View style={styles.dot} />
                <Text style={styles.statusLabel}>Live trading verified</Text>
              </View>
              <Text style={styles.meta}>Your Bitunix account is registered under TradeHub.</Text>
              {hasKeys ? (
                <Text style={[styles.meta, { color: colors.positive, marginTop: 4 }]}>
                  API keys connected — real orders will be placed when signals fire.
                </Text>
              ) : (
                <Text style={[styles.meta, { color: colors.warning, marginTop: 4 }]}>
                  No API keys yet — add them below to enable live order placement.
                </Text>
              )}
            </View>
          ) : notAffiliated ? (
            /* ── UID found but not under our affiliate ── */
            <View style={[styles.card, styles.cardWarn]}>
              <View style={styles.statusRow}>
                <Ionicons name="warning-outline" size={16} color={colors.warning} />
                <Text style={[styles.statusLabel, { color: colors.warning }]}>
                  Account not registered under TradeHub
                </Text>
              </View>
              <Text style={styles.meta}>
                Your Bitunix UID was found but your account wasn't signed up through the
                TradeHub affiliate link. Live trading is locked until you re-register.
              </Text>
              <Text style={[styles.meta, { marginTop: 6 }]}>
                To fix this: create a new Bitunix account using the link below (or contact
                Bitunix support to have your existing account moved to our IB).
              </Text>
              <Pressable style={styles.linkRow} onPress={openReferral}>
                <Ionicons name="open-outline" size={13} color={colors.positive} />
                <Text style={styles.linkText}>Sign up at Bitunix via TradeHub link</Text>
              </Pressable>
              <Pressable style={styles.copyRow} onPress={copyReferral}>
                <Ionicons name="copy-outline" size={13} color={colors.textDim} />
                <Text style={styles.copyText}>Copy link</Text>
              </Pressable>
            </View>
          ) : hasUid ? (
            /* ── UID set, check pending / API error ── */
            <View style={[styles.card, styles.cardNeutral]}>
              <View style={styles.statusRow}>
                <Ionicons name="time-outline" size={16} color={colors.textDim} />
                <Text style={[styles.statusLabel, { color: colors.textDim }]}>
                  Affiliate check pending
                </Text>
              </View>
              <Text style={styles.meta}>
                UID saved — verifying your account against the TradeHub affiliate roster.
                This can take up to 10 minutes on first connection.
              </Text>
            </View>
          ) : (
            /* ── No UID yet ── */
            <View style={[styles.card, styles.cardWarn]}>
              <View style={styles.statusRow}>
                <Ionicons name="alert-circle-outline" size={16} color={colors.warning} />
                <Text style={[styles.statusLabel, { color: colors.warning }]}>
                  Setup required — live trading locked
                </Text>
              </View>
              <Text style={styles.meta}>
                To unlock live crypto trading you need to:
              </Text>
              <Text style={styles.bulletItem}>① Register on Bitunix via the TradeHub link below</Text>
              <Text style={styles.bulletItem}>② Enter your Bitunix UID in the form below</Text>
              <Text style={styles.bulletItem}>③ Add your Bitunix API keys</Text>
            </View>
          )}

          {/* ── Referral link card (always shown until verified) ── */}
          {!isAffiliated && (
            <View style={styles.card}>
              <SectionLabel label="Step 1 — Register on Bitunix" />
              <Text style={styles.meta}>
                You must sign up through the TradeHub affiliate link to access live trading.
                This also gets you lower fees via the VIP code.
              </Text>
              <Pressable style={styles.referralBtn} onPress={openReferral}>
                <Ionicons name="open-outline" size={15} color="#000" />
                <Text style={styles.referralBtnText}>Register on Bitunix (VIP code applied)</Text>
              </Pressable>
              <Pressable style={styles.copyRow} onPress={copyReferral}>
                <Ionicons name="copy-outline" size={13} color={colors.textDim} />
                <Text style={styles.copyText}>{REFERRAL_URL}</Text>
              </Pressable>
            </View>
          )}

          {/* ── UID + API keys form ── */}
          <View style={styles.card}>
            <SectionLabel label={isAffiliated ? 'Update credentials' : 'Step 2 — Connect your account'} />

            <Text style={styles.fieldLabel}>Your Bitunix UID</Text>
            <Text style={styles.fieldHint}>
              Find it in Bitunix → Profile → UID. Looks like a long number.
            </Text>
            <TextInput
              style={styles.input}
              value={uidInput}
              onChangeText={setUidInput}
              placeholder={settings?.bitunix_uid ? `Current: ${settings.bitunix_uid}` : 'e.g. 1234567890'}
              placeholderTextColor={colors.textDim}
              keyboardType="numeric"
              autoCapitalize="none"
              autoCorrect={false}
            />

            <Pressable style={styles.toggleKeys} onPress={() => setShowKeys(v => !v)}>
              <Ionicons name={showKeys ? 'chevron-up' : 'chevron-down'} size={14} color={colors.textDim} />
              <Text style={styles.toggleKeysText}>
                {hasKeys ? 'Update API keys' : 'Add API keys'} (for live order placement)
              </Text>
            </Pressable>

            {showKeys && (
              <>
                <Text style={[styles.fieldLabel, { marginTop: spacing.sm }]}>API Key</Text>
                <Text style={styles.fieldHint}>
                  From Bitunix → Account → API Management. Enable "Trade" permission only.
                </Text>
                <TextInput
                  style={styles.input}
                  value={apiKey}
                  onChangeText={setApiKey}
                  placeholder="Paste your API key"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="none"
                  autoCorrect={false}
                  secureTextEntry
                />
                <Text style={[styles.fieldLabel, { marginTop: spacing.sm }]}>API Secret</Text>
                <TextInput
                  style={styles.input}
                  value={apiSecret}
                  onChangeText={setApiSecret}
                  placeholder="Paste your API secret"
                  placeholderTextColor={colors.textDim}
                  autoCapitalize="none"
                  autoCorrect={false}
                  secureTextEntry
                />
              </>
            )}

            <View style={{ marginTop: spacing.md }}>
              <PrimaryButton
                label={saveMut.isPending ? 'Saving…' : 'Save'}
                onPress={handleSave}
                disabled={saveMut.isPending}
              />
            </View>
          </View>

          {/* ── FP Markets IB note ── */}
          <View style={[styles.card, { borderColor: colors.border }]}>
            <SectionLabel label="Also trading Forex or Indices?" />
            <Text style={styles.meta}>
              For forex and index strategies, you'll also need an FP Markets account via
              the TradeHub introducing broker link — so your account is correctly linked to us.
            </Text>
            <Pressable style={styles.linkRow} onPress={openFPReferral}>
              <Ionicons name="open-outline" size={13} color={colors.positive} />
              <Text style={styles.linkText}>Open FP Markets account via TradeHub</Text>
            </Pressable>
            <Text style={[styles.meta, { marginTop: 8 }]}>
              After opening your account, go to{' '}
              <Text style={{ color: colors.text }}>Settings → Brokers → cTrader</Text>
              {' '}to connect via OAuth.
            </Text>
          </View>
        </>
      )}
    </Screen>
  );
}

const styles = StyleSheet.create({
  hero: {
    alignItems: 'center',
    paddingVertical: spacing.xl,
    paddingHorizontal: spacing.lg,
    marginBottom: spacing.md,
  },
  heroIcon: {
    width: 64,
    height: 64,
    borderRadius: 32,
    backgroundColor: `${colors.cardHi}`,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  heroIconOk: {
    backgroundColor: `${colors.positive}18`,
    borderColor: `${colors.positive}40`,
  },
  heroTitle: {
    fontSize: 20,
    fontFamily: font.bold,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  heroSub: {
    fontSize: 13,
    fontFamily: font.regular,
    color: colors.textDim,
    textAlign: 'center',
    lineHeight: 19,
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.lg,
    marginBottom: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  cardOk: {
    borderColor: `${colors.positive}40`,
    backgroundColor: `${colors.positive}08`,
  },
  cardWarn: {
    borderColor: `${colors.warning}40`,
    backgroundColor: `${colors.warning}08`,
  },
  cardNeutral: {
    borderColor: colors.border,
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: spacing.sm,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.positive,
  },
  statusLabel: {
    fontSize: 13,
    fontFamily: font.semibold,
    color: colors.positive,
    flex: 1,
  },
  meta: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 18,
    marginBottom: 2,
  },
  bulletItem: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    marginTop: 4,
    paddingLeft: 4,
  },
  linkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.sm,
  },
  linkText: {
    fontSize: 13,
    fontFamily: font.semibold,
    color: colors.positive,
  },
  copyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 6,
  },
  copyText: {
    fontSize: 11,
    fontFamily: font.regular,
    color: colors.textDim,
    flex: 1,
  },
  referralBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: colors.positive,
    borderRadius: radius.md,
    paddingVertical: 12,
    marginTop: spacing.md,
    marginBottom: spacing.sm,
  },
  referralBtnText: {
    fontSize: 14,
    fontFamily: font.semibold,
    color: '#000',
  },
  fieldLabel: {
    fontSize: 12,
    fontFamily: font.semibold,
    color: colors.text,
    marginBottom: 3,
    marginTop: spacing.sm,
  },
  fieldHint: {
    fontSize: 11,
    fontFamily: font.regular,
    color: colors.textDim,
    marginBottom: 6,
  },
  input: {
    backgroundColor: colors.cardHi,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 13,
    fontFamily: font.regular,
    color: colors.text,
  },
  toggleKeys: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.md,
    paddingVertical: 6,
  },
  toggleKeysText: {
    fontSize: 12,
    fontFamily: font.semibold,
    color: colors.textDim,
  },
});
