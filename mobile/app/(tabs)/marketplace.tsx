import React, { useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, ActivityIndicator, RefreshControl, Pressable } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { AmbientBg } from '@/components/AmbientBg';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type MarketplaceListing } from '@/lib/api';

function fmtPnl(v: number | null): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(1)}%`;
}

function ListingRow({ m, onPress }: { m: MarketplaceListing; onPress: () => void }) {
  const showLive = m.live_pnl !== null && m.live_trades >= 3;
  const headlinePnl = showLive ? m.live_pnl! : (m.verified_pnl ?? 0);
  const pnlColor = headlinePnl > 0 ? colors.positive : headlinePnl < 0 ? colors.negative : colors.text;
  const initial = (m.author_name || '?').charAt(0).toUpperCase();

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && { opacity: 0.85, transform: [{ scale: 0.998 }] }]}
    >
      <View style={styles.rowHeader}>
        <View style={styles.avatar}>
          <Text style={styles.avatarText}>{initial}</Text>
        </View>
        <View style={{ flex: 1 }}>
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

      <View style={styles.metaRow}>
        <View style={styles.metaItem}>
          <Text style={styles.metaLabel}>{showLive ? 'Live P&L' : 'Author P&L'}</Text>
          <Text style={[styles.metaValue, { color: pnlColor }]}>
            {showLive ? fmtPnl(m.live_pnl) : fmtPnl(m.verified_pnl)}
          </Text>
        </View>
        <View style={styles.metaSep} />
        <View style={styles.metaItem}>
          <Text style={styles.metaLabel}>Win</Text>
          <Text style={styles.metaValue}>
            {showLive ? `${m.live_win_rate?.toFixed(0) ?? '—'}%` : (m.verified_win_rate !== null ? `${m.verified_win_rate.toFixed(0)}%` : '—')}
          </Text>
        </View>
        <View style={styles.metaSep} />
        <View style={styles.metaItem}>
          <Text style={styles.metaLabel}>Clones</Text>
          <Text style={styles.metaValue}>{m.clone_count}</Text>
        </View>
        <View style={styles.metaSep} />
        <View style={styles.metaItem}>
          <Text style={styles.metaLabel}>Rating</Text>
          <Text style={styles.metaValue}>
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
  const router = useRouter();

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['marketplace', uid],
    queryFn: () => apiGet<MarketplaceListing[]>('/api/marketplace', uid, { sort: 'top' }),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  const Header = (
    <View style={styles.header}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
        <Ionicons name="storefront" size={22} color={colors.accent} />
        <Text style={styles.title}>Market</Text>
      </View>
      <Text style={styles.subtitle}>Top community strategies, ranked by performance.</Text>
    </View>
  );

  if (isLoading) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <AmbientBg variant="duo" />
        {Header}
        <View style={styles.center}><ActivityIndicator color={colors.accent} size="large" /></View>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <AmbientBg variant="duo" />
        {Header}
        <EmptyState icon="cloud-offline-outline" title="Couldn't load marketplace" hint="Pull down to retry." />
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <AmbientBg variant="duo" />
      <FlatList
        data={data || []}
        keyExtractor={(m) => `m-${m.id}`}
        renderItem={({ item }) => (
          <ListingRow m={item} onPress={() => router.push(`/listing/${item.id}` as any)} />
        )}
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
  title: { color: colors.text, fontFamily: font.black, fontSize: 30, letterSpacing: -0.8 },
  subtitle: { color: colors.textDim, fontFamily: font.regular, fontSize: 14, marginTop: 4 },
  listContent: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl + 96 },
  row: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
  },
  rowHeader: { flexDirection: 'row', alignItems: 'center', gap: spacing.md },
  avatar: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: colors.violetDim,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.32)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarText: {
    color: colors.violet,
    fontFamily: font.black,
    fontSize: 16,
  },
  rowTitle: { color: colors.text, fontFamily: font.bold, fontSize: 16, letterSpacing: -0.2 },
  rowAuthor: { color: colors.textDim, fontFamily: font.regular, fontSize: 12, marginTop: 2 },
  priceBox: {
    backgroundColor: colors.bgElev,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
  },
  priceFree: {
    color: colors.positive,
    fontFamily: font.black,
    fontSize: 11,
    letterSpacing: 0.7,
  },
  pricePaid: {
    color: colors.accent,
    fontFamily: font.black,
    fontSize: 13,
    fontVariant: ['tabular-nums'],
  },
  rowSummary: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    marginTop: spacing.md,
    lineHeight: 18,
  },
  badges: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: spacing.md },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  metaItem: { flex: 1, alignItems: 'center' },
  metaSep: { width: 1, height: 24, backgroundColor: colors.divider },
  metaLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 3,
  },
  metaValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
  },
});
