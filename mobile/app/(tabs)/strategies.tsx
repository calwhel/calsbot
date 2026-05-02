import React, { useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, Pressable, ActivityIndicator, RefreshControl } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { colors, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type Strategy } from '@/lib/api';

function statusTone(status: string): 'positive' | 'warning' | 'neutral' | 'negative' {
  if (status === 'active') return 'positive';
  if (status === 'paused') return 'warning';
  if (status === 'archived') return 'negative';
  return 'neutral';
}

function fmtPnl(v: number): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function StrategyRow({ s, onPress }: { s: Strategy; onPress: () => void }) {
  const perf = s.performance || {};
  const pnl = perf.total_pnl ?? 0;
  const trades = perf.total_trades ?? 0;
  const wr = perf.win_rate ?? 0;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [styles.row, pressed && { opacity: 0.7 }]}
    >
      <View style={styles.rowHeader}>
        <Text style={styles.rowTitle} numberOfLines={1}>{s.name}</Text>
        <Pill label={s.status} tone={statusTone(s.status)} small />
      </View>
      {s.description ? (
        <Text style={styles.rowDesc} numberOfLines={2}>{s.description}</Text>
      ) : null}
      <View style={styles.statRow}>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>P&amp;L</Text>
          <Text style={[
            styles.statValue,
            { color: pnl > 0 ? colors.positive : pnl < 0 ? colors.negative : colors.text },
          ]}>
            {trades > 0 ? fmtPnl(pnl) : '—'}
          </Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Win rate</Text>
          <Text style={styles.statValue}>{trades > 0 ? `${wr.toFixed(0)}%` : '—'}</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Trades</Text>
          <Text style={styles.statValue}>{trades}</Text>
        </View>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>Health</Text>
          <Text style={[
            styles.statValue,
            { color: s.health_score >= 7 ? colors.positive : s.health_score >= 4 ? colors.warning : colors.textDim },
          ]}>
            {s.health_score > 0 ? `${s.health_score.toFixed(1)}` : '—'}
          </Text>
        </View>
      </View>
    </Pressable>
  );
}

export default function StrategiesScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['strategies', uid],
    queryFn: () => apiGet<Strategy[]>('/api/strategies', uid),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  const renderItem = useCallback(
    ({ item }: { item: Strategy }) => (
      <StrategyRow s={item} onPress={() => router.push(`/strategy/${item.id}`)} />
    ),
    [router],
  );

  const Header = (
    <View style={styles.header}>
      <Text style={styles.title}>Strategies</Text>
      <Text style={styles.subtitle}>
        {data?.length ? `${data.length} strategy${data.length === 1 ? '' : ''}, tap any to inspect` : 'Your saved strategies'}
      </Text>
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
        <EmptyState icon="cloud-offline-outline" title="Couldn't load strategies" hint="Pull down to retry." />
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <FlatList
        data={data || []}
        keyExtractor={(s) => `s-${s.id}`}
        renderItem={renderItem}
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
            icon="rocket-outline"
            title="No strategies yet"
            hint="Build your first strategy on tradehub.markets, then come back here to track it on the go."
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
  rowHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  rowTitle: { color: colors.text, fontSize: 16, fontWeight: '700', flex: 1 },
  rowDesc: { color: colors.textDim, fontSize: 13, marginTop: 6, lineHeight: 18 },
  statRow: { flexDirection: 'row', marginTop: spacing.md, gap: spacing.lg },
  stat: { flex: 1 },
  statLabel: {
    color: colors.textMute, fontSize: 10, fontWeight: '700',
    letterSpacing: 0.6, textTransform: 'uppercase', marginBottom: 2,
  },
  statValue: { color: colors.text, fontSize: 15, fontWeight: '700' },
});
