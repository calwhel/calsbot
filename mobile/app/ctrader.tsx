import React from 'react';
import { View, Text, StyleSheet, Pressable, Alert, Linking, ActivityIndicator } from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { PrimaryButton } from '@/components/PrimaryButton';
import { SectionLabel } from '@/components/SectionLabel';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiDelete, ApiError } from '@/lib/api';

type CTraderStatus = {
  connected: boolean;
  account_id: string;
};

type AuthUrlResp = {
  url: string;
  redirect_uri: string;
};


export default function CTraderScreen() {
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();

  const { data: status, isLoading, refetch } = useQuery({
    queryKey: ['ctrader-status', uid],
    queryFn: () => apiGet<CTraderStatus>('/api/ctrader/status', uid),
    enabled: !!uid,
    refetchInterval: 5000,
  });

  const connect = useMutation({
    mutationFn: async () => {
      const data = await apiGet<AuthUrlResp>('/api/ctrader/auth-url', uid);
      if (!data?.url) throw new Error('No OAuth URL returned');
      return data;
    },
    onSuccess: async (data) => {
      await Haptics.selectionAsync();
      const supported = await Linking.canOpenURL(data.url);
      if (supported) {
        await Linking.openURL(data.url);
      } else {
        Alert.alert('Cannot open browser', `Visit this URL manually:\n${data.url}`);
      }
    },
    onError: (err: unknown) => {
      const msg = err instanceof ApiError ? err.message : 'Failed to start auth';
      Alert.alert('Connection error', msg);
    },
  });

  const disconnect = useMutation({
    mutationFn: () => apiDelete('/api/ctrader/disconnect', uid),
    onSuccess: async () => {
      await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      qc.invalidateQueries({ queryKey: ['ctrader-status', uid] });
      qc.invalidateQueries({ queryKey: ['portal-settings', uid] });
      Alert.alert('Disconnected', 'cTrader account disconnected. Forex strategies will run in paper mode.');
    },
    onError: () => Alert.alert('Error', 'Failed to disconnect.'),
  });

  const handleDisconnect = () => {
    Alert.alert(
      'Disconnect cTrader?',
      'Forex live strategies will revert to paper mode until you reconnect.',
      [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Disconnect', style: 'destructive', onPress: () => disconnect.mutate() },
      ],
    );
  };

  return (
    <Screen title="cTrader — FP Markets" scroll>
      <Stack.Screen options={{ title: 'cTrader — FP Markets' }} />

      {/* Hero */}
      <View style={styles.hero}>
        <View style={styles.heroIcon}>
          <Ionicons name="globe-outline" size={32} color={colors.positive} />
        </View>
        <Text style={styles.heroTitle}>FP Markets via cTrader</Text>
        <Text style={styles.heroSub}>
          Connect your FP Markets live account to run forex strategies with real execution.
          Paper trading runs automatically without any connection.
        </Text>
      </View>

      {isLoading ? (
        <ActivityIndicator color={colors.positive} style={{ marginTop: spacing.xl }} />
      ) : status?.connected ? (
        /* ── Connected state ── */
        <View style={styles.card}>
          <View style={styles.connectedRow}>
            <View style={styles.dot} />
            <Text style={styles.connectedLabel}>cTrader connected — live forex active</Text>
          </View>
          <Text style={styles.meta}>
            Account ID: <Text style={styles.mono}>{status.account_id || '—'}</Text>
          </Text>
          <Text style={styles.meta}>Broker: FP Markets · Live</Text>

          <Pressable
            style={styles.disconnectBtn}
            onPress={handleDisconnect}
            disabled={disconnect.isPending}
          >
            <Text style={styles.disconnectText}>
              {disconnect.isPending ? 'Disconnecting…' : 'Disconnect'}
            </Text>
          </Pressable>
        </View>
      ) : (
        /* ── Not connected state ── */
        <View>
          <View style={styles.card}>
            <SectionLabel label="How it works" />
            <View style={styles.step}>
              <Text style={styles.stepNum}>1</Text>
              <Text style={styles.stepText}>
                Tap <Text style={styles.bold}>Connect via cTrader</Text> below. You'll be taken to
                Spotware's login page in your browser.
              </Text>
            </View>
            <View style={styles.step}>
              <Text style={styles.stepNum}>2</Text>
              <Text style={styles.stepText}>
                Log in with your <Text style={styles.bold}>FP Markets cTrader ID</Text> and
                authorise TradeHub to place trades on your behalf.
              </Text>
            </View>
            <View style={styles.step}>
              <Text style={styles.stepNum}>3</Text>
              <Text style={styles.stepText}>
                You'll be redirected back automatically. Forex strategies set to{' '}
                <Text style={styles.bold}>Go Live</Text> will start routing real orders through
                your FP Markets account.
              </Text>
            </View>
          </View>

          <View style={styles.warningCard}>
            <Ionicons name="warning-outline" size={16} color={colors.warning} />
            <Text style={styles.warningText}>
              Only connect a <Text style={styles.bold}>live account</Text> you're comfortable
              using. Paper testing works on our platform without any connection.
            </Text>
          </View>

          <View style={{ marginTop: spacing.lg }}>
            <PrimaryButton
              label={connect.isPending ? 'Opening browser…' : 'Connect via cTrader OAuth'}
              onPress={() => connect.mutate()}
              disabled={connect.isPending}
            />
          </View>

          <Pressable style={styles.refreshBtn} onPress={() => refetch()}>
            <Ionicons name="refresh-outline" size={14} color={colors.textDim} />
            <Text style={styles.refreshText}>Refresh connection status</Text>
          </Pressable>
        </View>
      )}

      {/* Info footer */}
      <View style={styles.footer}>
        <Text style={styles.footerText}>
          Don't have an FP Markets account?{' '}
          <Text
            style={styles.link}
            onPress={() => Linking.openURL('https://www.fpmarkets.com/?fpm-affiliate-utm-source=IB&fpm-affiliate-model=revenue-sharing')}
          >
            Open one via TradeHub's IB link
          </Text>
          {' '}— choose the cTrader platform when signing up. Your account must be opened through our introducing broker link to be eligible for live trading on TradeHub.
        </Text>
      </View>
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
    backgroundColor: `${colors.positive}18`,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
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
  connectedRow: {
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
  connectedLabel: {
    fontSize: 13,
    fontFamily: font.semibold,
    color: colors.positive,
    flex: 1,
  },
  meta: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    marginBottom: 3,
  },
  mono: {
    fontFamily: 'monospace',
    color: colors.text,
  },
  disconnectBtn: {
    marginTop: spacing.md,
    paddingVertical: 8,
    paddingHorizontal: 14,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.negative,
    alignSelf: 'flex-start',
  },
  disconnectText: {
    fontSize: 12,
    fontFamily: font.semibold,
    color: colors.negative,
  },
  step: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: spacing.sm,
    marginTop: spacing.xs,
  },
  stepNum: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.cardHi,
    textAlign: 'center',
    lineHeight: 22,
    fontSize: 12,
    fontFamily: font.bold,
    color: colors.positive,
    overflow: 'hidden',
  },
  stepText: {
    flex: 1,
    fontSize: 13,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 19,
  },
  bold: {
    fontFamily: font.semibold,
    color: colors.text,
  },
  warningCard: {
    flexDirection: 'row',
    gap: 10,
    backgroundColor: `${colors.warning}12`,
    borderWidth: 1,
    borderColor: `${colors.warning}30`,
    borderRadius: radius.md,
    padding: spacing.md,
    marginBottom: spacing.sm,
    alignItems: 'flex-start',
  },
  warningText: {
    flex: 1,
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 18,
  },
  refreshBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    justifyContent: 'center',
    marginTop: spacing.md,
    padding: spacing.sm,
  },
  refreshText: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
  },
  footer: {
    marginTop: spacing.xl,
    paddingTop: spacing.lg,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  footerText: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 18,
    textAlign: 'center',
  },
  link: {
    color: colors.positive,
    fontFamily: font.semibold,
  },
});
