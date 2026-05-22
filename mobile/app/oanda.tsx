import React, { useCallback, useState } from 'react';
import { View, Text, StyleSheet, TextInput, Pressable, Alert, Platform, Linking, ActivityIndicator } from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { PrimaryButton } from '@/components/PrimaryButton';
import { SectionLabel } from '@/components/SectionLabel';
import { Pill } from '@/components/Pill';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPost, apiDelete, ApiError } from '@/lib/api';

type OandaStatus = {
  connected: boolean;
  environment: 'practice' | 'live';
  account_id: string;
};

type OandaConnectResp = {
  ok: boolean;
  account?: {
    id: string;
    currency: string;
    balance: number;
    nav: number;
    open_position_count: number;
    margin_available: number;
  };
};

export default function OandaScreen() {
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();

  const [apiKey, setApiKey]         = useState('');
  const [accountId, setAccountId]   = useState('');
  const [environment, setEnvironment] = useState<'practice' | 'live'>('practice');
  const [lastSummary, setLastSummary] = useState<OandaConnectResp['account'] | null>(null);

  const { data: status, isLoading } = useQuery({
    queryKey: ['oanda-status', uid],
    queryFn: () => apiGet<OandaStatus>('/api/oanda/status', uid),
    enabled: !!uid,
  });

  const connect = useMutation({
    mutationFn: () =>
      apiPost<OandaConnectResp>('/api/oanda/connect', {
        api_key: apiKey.trim(),
        account_id: accountId.trim(),
        environment,
      }, uid),
    onSuccess: (resp) => {
      setLastSummary(resp.account || null);
      setApiKey('');
      qc.invalidateQueries({ queryKey: ['oanda-status', uid] });
      qc.invalidateQueries({ queryKey: ['settings', uid] });
      if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      Alert.alert('Connected', `OANDA ${environment} account linked. Balance: ${resp.account?.balance?.toFixed(2)} ${resp.account?.currency}`);
    },
    onError: (e: unknown) => {
      const msg = e instanceof ApiError ? e.message : 'Could not validate OANDA credentials.';
      Alert.alert('Connection failed', msg);
    },
  });

  const disconnect = useMutation({
    mutationFn: () => apiDelete<{ ok: boolean }>('/api/oanda/disconnect', uid),
    onSuccess: () => {
      setLastSummary(null);
      qc.invalidateQueries({ queryKey: ['oanda-status', uid] });
      qc.invalidateQueries({ queryKey: ['settings', uid] });
      if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    },
    onError: (e: unknown) => {
      const msg = e instanceof ApiError ? e.message : 'Could not disconnect OANDA.';
      Alert.alert('Disconnect failed', msg);
    },
  });

  const onDisconnect = useCallback(() => {
    Alert.alert(
      'Disconnect OANDA?',
      'Forex strategies will fall back to paper mode until you reconnect.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Disconnect', style: 'destructive', onPress: () => disconnect.mutate() },
      ],
    );
  }, [disconnect]);

  const canSubmit = apiKey.trim().length > 20 && accountId.trim().length > 4;

  return (
    <>
      <Stack.Screen options={{ title: 'OANDA Broker', headerStyle: { backgroundColor: colors.bg }, headerTintColor: colors.text }} />
      <Screen title="Connect OANDA" subtitle="Link your forex broker to enable live forex execution." ambient="violet">
        {isLoading ? (
          <ActivityIndicator color={colors.accent} style={{ marginTop: spacing.xl }} />
        ) : status?.connected ? (
          <View style={styles.card}>
            <View style={styles.rowBetween}>
              <Text style={styles.cardTitle}>Connected</Text>
              <Pill label={status.environment === 'live' ? 'LIVE' : 'PRACTICE'} tone={status.environment === 'live' ? 'negative' : 'positive'} small />
            </View>
            <Text style={styles.cardHint}>Account ID</Text>
            <Text style={styles.cardValue}>{status.account_id}</Text>
            {lastSummary ? (
              <>
                <View style={styles.divider} />
                <View style={styles.statRow}>
                  <View style={styles.statCol}>
                    <Text style={styles.statLabel}>Balance</Text>
                    <Text style={styles.statValue}>{lastSummary.balance.toFixed(2)} {lastSummary.currency}</Text>
                  </View>
                  <View style={styles.statCol}>
                    <Text style={styles.statLabel}>Open positions</Text>
                    <Text style={styles.statValue}>{lastSummary.open_position_count}</Text>
                  </View>
                </View>
              </>
            ) : null}
            <View style={{ height: spacing.lg }} />
            <PrimaryButton label="Disconnect" variant="destructive" onPress={onDisconnect} loading={disconnect.isPending} />
          </View>
        ) : (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>OANDA credentials</Text>
            <Text style={styles.cardHint}>
              Generate a personal access token at developer.oanda.com → My Account → Manage API Access. You also need your account ID from the OANDA dashboard.
            </Text>

            <SectionLabel label="Environment" />
            <View style={styles.envRow}>
              {(['practice', 'live'] as const).map((env) => (
                <Pressable
                  key={env}
                  onPress={() => setEnvironment(env)}
                  style={[styles.envChip, environment === env && styles.envChipActive]}
                >
                  <Text style={[styles.envChipText, environment === env && styles.envChipTextActive]}>
                    {env === 'practice' ? 'Practice (recommended)' : 'Live'}
                  </Text>
                </Pressable>
              ))}
            </View>

            <SectionLabel label="API key" />
            <TextInput
              value={apiKey}
              onChangeText={setApiKey}
              placeholder="Personal access token"
              placeholderTextColor={colors.textDim}
              autoCapitalize="none"
              autoCorrect={false}
              secureTextEntry
              style={styles.input}
            />

            <SectionLabel label="Account ID" />
            <TextInput
              value={accountId}
              onChangeText={setAccountId}
              placeholder="e.g. 101-001-12345678-001"
              placeholderTextColor={colors.textDim}
              autoCapitalize="none"
              autoCorrect={false}
              style={styles.input}
            />

            <View style={{ height: spacing.lg }} />
            <PrimaryButton
              label={connect.isPending ? 'Validating...' : 'Connect'}
              onPress={() => connect.mutate()}
              disabled={!canSubmit || connect.isPending}
              loading={connect.isPending}
            />

            <Pressable onPress={() => Linking.openURL('https://developer.oanda.com/rest-live-v20/introduction/').catch(() => {})} style={styles.linkRow}>
              <Ionicons name="open-outline" size={14} color={colors.textDim} />
              <Text style={styles.linkText}>OANDA API docs</Text>
            </Pressable>
          </View>
        )}
      </Screen>
    </>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    marginTop: spacing.md,
  },
  cardTitle:  { fontFamily: font.semibold, fontSize: 18, color: colors.text, marginBottom: spacing.xs },
  cardHint:   { fontFamily: font.regular, fontSize: 13, color: colors.textDim, marginBottom: spacing.md, lineHeight: 18 },
  cardValue:  { fontFamily: font.semibold, fontSize: 15, color: colors.text, marginBottom: spacing.sm },
  rowBetween: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.sm },
  divider:    { height: 1, backgroundColor: colors.border, marginVertical: spacing.md },
  statRow:    { flexDirection: 'row', gap: spacing.lg },
  statCol:    { flex: 1 },
  statLabel:  { fontFamily: font.regular, fontSize: 11, color: colors.textDim, textTransform: 'uppercase', letterSpacing: 0.5 },
  statValue:  { fontFamily: font.semibold, fontSize: 16, color: colors.text, marginTop: 4 },
  envRow:     { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  envChip:    { flex: 1, paddingVertical: spacing.sm, paddingHorizontal: spacing.md, borderRadius: radius.md, borderWidth: 1, borderColor: colors.border, alignItems: 'center', backgroundColor: colors.cardHi },
  envChipActive: { borderColor: colors.accent, backgroundColor: colors.accent + '22' },
  envChipText:   { fontFamily: font.semibold, fontSize: 13, color: colors.textDim },
  envChipTextActive: { color: colors.text },
  input: {
    backgroundColor: colors.cardHi,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm + 2,
    fontFamily: font.regular,
    fontSize: 14,
    color: colors.text,
    marginBottom: spacing.md,
  },
  linkRow:   { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 6, marginTop: spacing.md },
  linkText:  { fontFamily: font.regular, fontSize: 12, color: colors.textDim, textDecorationLine: 'underline' },
});
