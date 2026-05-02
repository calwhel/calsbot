import React, { useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, ActivityIndicator, RefreshControl, Pressable } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { colors, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type MarketplaceListing } from '@/lib/api';

function fmtPnl(v: number | null): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(1)}%`;
}

function ListingRow({ m }: { m: MarketplaceListing }) {
  const showLive = m.live_pnl !== null && m.live_trades >= 3;
  return (
    <Pressable style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]}>
      <View style={styles.rowHeader}>
        <View style={{ flex: 1, marginRight: spacing.sm }}>
          <Text style={styles.rowTitle} numberOfLines={1}>{m.title}</Text>
          <Text style={styles.rowAuthor} numberOfLines={1}>by {m.author_name}</Text>
        </View>
        <View style={styles.priceBox}>
          {m.pricing_model === 'free' ? (
            <Text style={styles.priceFree}>FREE</Text>
          ) : (
            <Text style={styles.pricePaid}>${m.price_usdt}</Text>
          )}
        </View>
      </View>

      {m.summary ? (
        <Text style={styles.rowSummary} numberOfLines={2}>{m.summary}</Text>
      ) : null}

      <View style={styles.badges}>
        {m.is_verified ? <Pill label="✓ Verified" tone="accent" small /> : null}
        {m.is_trending ? <Pill label="Trending" tone="warning" small /> : null}
        {m.is_featured ? <Pill label="Featured" tone="positive" small /> : null}
        <Pill label={m.category} tone="neutral" small />
      </View>

      <View style={styles.statRow}>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>{showLive ? 'Live P&L' : 'Author P&L'}</Text>
          <Text style={[
            styles.statValue,
            { color: (showLive ? m.live_pnl! : (m.verified_pnl ?? 0)) >= 0 ? colors.positive : colors.negative },
          ]}>
            {showLive ? fmtPnl(m.live_pnl) : fmtPnl(m.verified_pnl)}
          </Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Win rate</Text>
          <Text style={styles.statValue}>
            {showLive ? `${m.live_win_rate?.toFixed(0) ?? '—'}%` : (m.verified_win_rate !== null ? `${m.verified_win_rate.toFixed(0)}%` : '—')}
          </Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Clones</Text>
          <Text style={styles.statValue}>{m.clone_count}</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Rating</Text>
          <Text style={styles.statValue}>
            {m.rating_count > 0 ? `${m.avg_rating.toFixed(1)} ★` : '—'}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

export default function MarketplaceScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['marketplace', uid],
    queryFn: () => apiGet<MarketplaceListing[]>('/api/marketplace', uid, { sort: 'top' }),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  const Header = (
    <View style={styles.header}>
      <Text style={styles.title}>Market</Text>
      <Text style={styles.subtitle}>Top community strategies, ranked by performance.</Text>
    </View>
  );

  if (isLoading) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        {Header}
        <View style={styles.center}><ActivityIndicator color={colors.accent} size="large" /></View>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        {Header}
        <EmptyState icon="cloud-offline-outline" title="Couldn't load marketplace" hint="Pull down to retry." />
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <FlatList
        data={data || []}
        keyExtractor={(m) => `m-${m.id}`}
        renderItem={({ item }) => <ListingRow m={item} />}
        ListHeaderComponent={Header}
        ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
        contentContainerStyle={styles.listContent}
        refreshControl={
          <RefreshControl
            refreshing={isFetching && !isLoading}
            onRefresh={onRefresh}
            tintColor={colors.accent}
            colors={[colors.accent]}
            progressBackgroundColor={colors.bgElev}
          />
        }
        ListEmptyComponent={
          <EmptyState
            icon="storefront-outline"
            title="Marketplace is quiet"
            hint="Be the first to publish a strategy from the web portal."
          />
        }
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.lg },
  title: { color: colors.text, fontSize: 28, fontWeight: '800', letterSpacing: -0.5 },
  subtitle: { color: colors.textDim, fontSize: 14, marginTop: 2 },
  listContent: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl + 80 },
  row: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
  },
  rowHeader: { flexDirection: 'row', alignItems: 'flex-start' },
  rowTitle: { color: colors.text, fontSize: 16, fontWeight: '700' },
  rowAuthor: { color: colors.textDim, fontSize: 12, marginTop: 2 },
  priceBox: {
    backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.border,
    paddingHorizontal: 10, paddingVertical: 5,
    borderRadius: radius.sm,
  },
  priceFree: { color: colors.positive, fontSize: 12, fontWeight: '800', letterSpacing: 0.6 },
  pricePaid: { color: colors.accent, fontSize: 13, fontWeight: '800' },
  rowSummary: { color: colors.textDim, fontSize: 13, marginTop: 8, lineHeight: 18 },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: spacing.md },
  statRow: { flexDirection: 'row', marginTop: spacing.md, gap: spacing.md },
  stat: { flex: 1 },
  statLabel: {
    color: colors.textMute, fontSize: 10, fontWeight: '700',
    letterSpacing: 0.5, textTransform: 'uppercase', marginBottom: 2,
  },
  statValue: { color: colors.text, fontSize: 14, fontWeight: '700' },
});
