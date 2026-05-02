import React, { useCallback } from 'react';
import {
  View, Text, StyleSheet, ScrollView, ActivityIndicator,
  Alert, Pressable, Linking, useWindowDimensions,
} from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { StatCard } from '@/components/StatCard';
import { PrimaryButton } from '@/components/PrimaryButton';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPost, ApiError, getApiUrl, type MarketplaceListingDetail, type CloneResponse } from '@/lib/api';
// Note: `user` is intentionally unread here — entitlement gating is delegated to
// the backend's structured PRO_REQUIRED response so a stale local is_pro flag
// can never block a freshly-upgraded user.

function fmtPnl(v: number | null | undefined): string {
  if (v == null) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

export default function ListingDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const lid = Number(id);
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();
  const { width } = useWindowDimensions();

  const detailQ = useQuery({
    queryKey: ['marketplace-detail', lid, uid],
    queryFn: () => apiGet<MarketplaceListingDetail>(`/api/marketplace/${lid}`, uid),
    enabled: !!uid && !!lid,
  });

  const cloneM = useMutation({
    mutationFn: () => apiPost<CloneResponse>(`/api/marketplace/${lid}/clone`, {}, uid),
    onSuccess: (resp) => {
      // Backend returns several 200-with-shape responses we must distinguish:
      //   { error: "PRO_REQUIRED", message } — Pro upsell
      //   { requires_payment: true, price_usdt, pricing_model, message } — paid listing
      //   { already_owned: true, cloned_strategy_id } — user already has it
      //   { success: true, strategy_id } — free clone succeeded
      // 403-with-error is handled in onError below.
      if (resp.error === 'PRO_REQUIRED') {
        promptUpgrade(resp.message || 'A Pro subscription is required.');
        return;
      }
      if (resp.requires_payment) {
        promptPayment(resp.price_usdt ?? 0, resp.message);
        return;
      }
      if (resp.already_owned) {
        qc.invalidateQueries({ queryKey: ['marketplace-detail', lid, uid] });
        Alert.alert(
          'Already in your library',
          'You already have this strategy. Open it from the Strategies tab.',
          [
            { text: 'OK', style: 'cancel' },
            {
              text: 'View strategies',
              onPress: () => router.replace('/(tabs)/strategies' as any),
            },
          ],
        );
        return;
      }
      // Free-clone happy path (success === true || strategy_id present).
      qc.invalidateQueries({ queryKey: ['strategies', uid] });
      qc.invalidateQueries({ queryKey: ['marketplace-detail', lid, uid] });
      qc.invalidateQueries({ queryKey: ['marketplace', uid] });
      Alert.alert(
        'Strategy added',
        'The strategy is now in your portfolio in paper-trading mode. Activate it from the Strategies tab when you’re ready.',
        [
          { text: 'Stay here', style: 'cancel' },
          {
            text: 'View strategies',
            onPress: () => router.replace('/(tabs)/strategies' as any),
          },
        ],
      );
    },
    onError: (e) => {
      // 403 PRO_REQUIRED comes through here as ApiError with the JSON-encoded body.
      if (e instanceof ApiError && e.status === 403 && e.body) {
        try {
          const j = JSON.parse(e.body);
          if (j?.error === 'PRO_REQUIRED') {
            promptUpgrade(j.message || 'A Pro subscription is required.');
            return;
          }
        } catch {}
      }
      Alert.alert('Could not add strategy', (e as Error).message || 'Try again later.');
    },
  });

  const promptUpgrade = useCallback((msg: string) => {
    Alert.alert(
      'Pro required',
      msg,
      [
        { text: 'Not now', style: 'cancel' },
        {
          text: 'Open billing',
          onPress: () => {
            const base = getApiUrl().replace(/\/api\/?$/, '');
            Linking.openURL(`${base}/billing`).catch(() => {
              Alert.alert('Could not open browser', 'Visit tradehub.markets in your browser to upgrade.');
            });
          },
        },
      ],
    );
  }, []);

  const promptPayment = useCallback((priceUsdt: number, msg?: string) => {
    Alert.alert(
      'Payment required',
      msg || `This strategy costs $${priceUsdt.toFixed(2)}. Complete the purchase on the web to add it to your library.`,
      [
        { text: 'Not now', style: 'cancel' },
        {
          text: 'Open in browser',
          onPress: () => {
            const base = getApiUrl().replace(/\/api\/?$/, '');
            // Marketplace listing detail page handles checkout via Telegram/OxaPay.
            Linking.openURL(`${base}/marketplace?listing=${lid}`).catch(() => {
              Alert.alert('Could not open browser', 'Visit tradehub.markets in your browser to complete the purchase.');
            });
          },
        },
      ],
    );
  }, [lid]);

  if (detailQ.isLoading) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <View style={[styles.center, { backgroundColor: colors.bg, flex: 1 }]}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      </>
    );
  }

  if (detailQ.isError || !detailQ.data) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <EmptyState icon="alert-circle-outline" title="Listing unavailable" hint="It may have been removed by the author." />
      </>
    );
  }

  const m = detailQ.data;
  const isPaid = m.pricing_model !== 'free' && (m.price_usdt || 0) > 0;
  const owned = m.is_owned;
  // We intentionally do NOT preflight a Pro-required block here — the backend
  // is the source of truth for entitlement. Cached `user.is_pro` can be stale
  // (e.g. user just upgraded on web), and the backend returns a structured
  // PRO_REQUIRED response that we handle below in onSuccess/onError.

  return (
    <>
      <Stack.Screen options={{ title: '' }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: colors.bg }}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 32 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title} numberOfLines={3}>{m.title}</Text>
          <Text style={styles.author}>by {m.author_name}</Text>
          <View style={styles.headerMeta}>
            {m.is_verified ? <Pill label="✓ Verified" tone="accent" small /> : null}
            <Pill label={m.category} tone="neutral" small />
            {isPaid ? (
              <Pill label={`$${m.price_usdt}`} tone="warning" small />
            ) : (
              <Pill label="FREE" tone="positive" small />
            )}
            {owned ? <Pill label="In your library" tone="positive" small /> : null}
          </View>
          {m.summary ? (
            <Text style={styles.summary}>{m.summary}</Text>
          ) : null}
        </View>

        {/* Action */}
        <View style={{ marginTop: spacing.lg }}>
          {owned ? (
            <View style={[styles.bannerOwned]}>
              <Ionicons name="checkmark-circle" size={20} color={colors.positive} />
              <Text style={styles.bannerText}>You already have this strategy.</Text>
            </View>
          ) : (
            <PrimaryButton
              label={
                isPaid
                  ? `Buy & clone — $${m.price_usdt}`
                  : 'Clone to my strategies'
              }
              onPress={() => cloneM.mutate()}
              loading={cloneM.isPending}
            />
          )}
          {!owned ? (
            <Text style={styles.helpText}>
              Cloned strategies start in paper-trading mode. You can review and activate
              them from the Strategies tab.
            </Text>
          ) : null}
        </View>

        {/* Live performance */}
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.sectionLabel}>Live performance</Text>
          <View style={styles.statRow}>
            <StatCard
              label="Total P&L"
              value={m.live_performance.total_trades > 0 ? fmtPnl(m.live_performance.total_pnl) : '—'}
              tone={m.live_performance.total_pnl > 0 ? 'positive' : m.live_performance.total_pnl < 0 ? 'negative' : 'neutral'}
            />
            <View style={{ width: spacing.md }} />
            <StatCard
              label="Win rate"
              value={m.live_performance.total_trades > 0 ? `${m.live_performance.win_rate.toFixed(1)}%` : '—'}
              sub={`${m.live_performance.total_trades} closed`}
            />
          </View>
          {m.is_verified ? (
            <View style={[styles.statRow, { marginTop: spacing.md }]}>
              <StatCard
                label="Verified P&L"
                value={fmtPnl(m.live_performance.total_pnl)}
                tone="positive"
                compact
              />
              <View style={{ width: spacing.md }} />
              <StatCard
                label="Verified WR"
                value={`${m.verified_win_rate.toFixed(1)}%`}
                sub={`${m.verified_trades} trades`}
                compact
              />
              <View style={{ width: spacing.md }} />
              <StatCard
                label="Clones"
                value={`${m.clone_count}`}
                compact
              />
            </View>
          ) : null}
        </View>

        {/* Recent trades */}
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.sectionLabel}>Recent trades</Text>
          {m.recent_trades.length === 0 ? (
            <EmptyState icon="hourglass-outline" title="No trades yet" hint="The author hasn’t fired any signals yet." />
          ) : (
            <View style={styles.tradeList}>
              {m.recent_trades.map((t, i) => (
                <View key={`rt-${i}`} style={[styles.tradeRow, i < m.recent_trades.length - 1 && styles.tradeRowDiv]}>
                  <View style={{ flex: 1 }}>
                    <Text style={styles.tradeSymbol}>{t.symbol}</Text>
                    <View style={styles.tradeMeta}>
                      <Pill label={t.direction} tone={t.direction === 'LONG' ? 'positive' : 'negative'} small />
                      <Text style={styles.tradeOutcome}>{t.outcome}</Text>
                    </View>
                  </View>
                  <Text style={[
                    styles.tradePnl,
                    {
                      color: (t.pnl_pct ?? 0) > 0 ? colors.positive
                        : (t.pnl_pct ?? 0) < 0 ? colors.negative
                        : colors.textDim,
                    },
                  ]}>
                    {fmtPnl(t.pnl_pct)}
                  </Text>
                </View>
              ))}
            </View>
          )}
        </View>

        {/* Ratings */}
        {m.rating_count > 0 ? (
          <View style={{ marginTop: spacing.lg }}>
            <Text style={styles.sectionLabel}>
              Ratings · {m.avg_rating.toFixed(1)} ★ ({m.rating_count})
            </Text>
            <View style={styles.tradeList}>
              {m.ratings.slice(0, 5).map((r, i) => (
                <View key={`r-${i}`} style={[styles.tradeRow, i < Math.min(m.ratings.length, 5) - 1 && styles.tradeRowDiv, { flexDirection: 'column', alignItems: 'flex-start' }]}>
                  <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                    <Text style={{ color: colors.warning, fontSize: 14, fontWeight: '700' }}>
                      {'★'.repeat(r.stars)}{'☆'.repeat(5 - r.stars)}
                    </Text>
                    {r.is_verified ? <Pill label="verified" tone="accent" small /> : null}
                  </View>
                  {r.review ? (
                    <Text style={{ color: colors.textDim, fontSize: 13, marginTop: 6, lineHeight: 18 }}>
                      {r.review}
                    </Text>
                  ) : null}
                </View>
              ))}
            </View>
          </View>
        ) : null}
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.xxl + 96 },
  center: { alignItems: 'center', justifyContent: 'center', flex: 1, padding: spacing.xl },
  header: { paddingTop: spacing.sm },
  title: { color: colors.text, fontFamily: font.black, fontSize: 26, letterSpacing: -0.6 },
  author: { color: colors.textDim, fontFamily: font.medium, fontSize: 14, marginTop: 4 },
  headerMeta: { flexDirection: 'row', gap: 6, marginTop: spacing.sm, flexWrap: 'wrap' },
  summary: { color: colors.textDim, fontFamily: font.regular, fontSize: 14, marginTop: spacing.md, lineHeight: 20 },
  sectionLabel: {
    color: colors.textDim, fontFamily: font.bold, fontSize: 12,
    letterSpacing: 0.6, textTransform: 'uppercase', marginBottom: spacing.sm,
  },
  statRow: { flexDirection: 'row' },
  helpText: {
    color: colors.textMute, fontFamily: font.regular, fontSize: 12, marginTop: spacing.sm,
    lineHeight: 17, textAlign: 'center',
  },
  bannerOwned: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    backgroundColor: colors.positiveDim,
    borderWidth: 1, borderColor: 'rgba(52,211,153,0.32)', borderRadius: radius.lg,
    paddingHorizontal: spacing.lg, paddingVertical: spacing.md,
  },
  bannerText: { color: colors.text, fontFamily: font.semibold, fontSize: 14 },
  tradeList: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1, borderColor: colors.border,
    overflow: 'hidden',
  },
  tradeRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: spacing.lg, paddingVertical: 12,
  },
  tradeRowDiv: { borderBottomWidth: 1, borderBottomColor: colors.divider },
  tradeSymbol: { color: colors.text, fontFamily: font.bold, fontSize: 15 },
  tradeMeta: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 },
  tradeOutcome: { color: colors.textMute, fontFamily: font.bold, fontSize: 11, letterSpacing: 0.5 },
  tradePnl: { fontFamily: font.bold, fontSize: 15, fontVariant: ['tabular-nums'] },
});
