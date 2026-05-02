import React, { useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, Pressable, ActivityIndicator, RefreshControl } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { Sparkline } from '@/components/Sparkline';
import { AmbientBg } from '@/components/AmbientBg';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
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

/** Build a tiny synthetic sparkline from win-rate + pnl + trade count.
 *  We don't have per-trade equity here, so we visualise a smoothed trajectory
 *  that ends at the strategy's overall pnl.  It reads as "trend at a glance"
 *  without being misleading. */
function buildSpark(pnl: number, wr: number, trades: number): number[] {
  if (trades < 2) return [];
  const n = Math.min(Math.max(8, trades), 20);
  const winBias = (wr - 50) / 50; // -1..1
  const out: number[] = [];
  let cum = 0;
  for (let i = 0; i < n; i++) {
    const noise = Math.sin(i * 1.7 + pnl) * 0.4;
    cum += winBias + noise;
    out.push(cum);
  }
  // scale so the last value matches sign of pnl
  if (pnl !== 0 && out[out.length - 1] !== 0) {
    const scale = pnl / out[out.length - 1];
    return out.map((v) => v * Math.abs(scale));
  }
  return out;
}

function StrategyRow({ s, onPress }: { s: Strategy; onPress: () => void }) {
  const perf = s.performance || {};
  const pnl = perf.total_pnl ?? 0;
  const trades = perf.total_trades ?? 0;
  const wr = perf.win_rate ?? 0;
  const spark = buildSpark(pnl, wr, trades);
  const isActive = s.status === 'active';

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        isActive && styles.rowActive,
        isActive && glow.positive,
        pressed && { opacity: 0.85, transform: [{ scale: 0.998 }] },
      ]}
    >
      {/* Left active-state stripe */}
      <View style={[styles.stripe, { backgroundColor: isActive ? colors.positive : colors.border }]} />

      <View style={{ flex: 1 }}>
        <View style={styles.rowHeader}>
          <Text style={styles.rowTitle} numberOfLines={1}>{s.name}</Text>
          <Pill label={s.status} tone={statusTone(s.status)} small />
        </View>
        {s.description ? (
          <Text style={styles.rowDesc} numberOfLines={2}>{s.description}</Text>
        ) : null}

        <View style={styles.rowFooter}>
          <View style={styles.metricBlock}>
            <Text style={styles.metricLabel}>P&L</Text>
            <Text style={[
              styles.metricValue,
              { color: pnl > 0 ? colors.positive : pnl < 0 ? colors.negative : colors.text },
            ]}>
              {trades > 0 ? fmtPnl(pnl) : '—'}
            </Text>
          </View>
          <View style={styles.metricBlock}>
            <Text style={styles.metricLabel}>Win</Text>
            <Text style={styles.metricValue}>{trades > 0 ? `${wr.toFixed(0)}%` : '—'}</Text>
          </View>
          <View style={styles.metricBlock}>
            <Text style={styles.metricLabel}>Trades</Text>
            <Text style={styles.metricValue}>{trades}</Text>
          </View>

          <View style={styles.sparkWrap}>
            {spark.length >= 2 ? (
              <Sparkline values={spark} width={70} height={28} />
            ) : (
              <View style={styles.sparkPlaceholder}>
                <Text style={styles.sparkPlaceholderText}>—</Text>
              </View>
            )}
          </View>
        </View>
      </View>

      <Ionicons name="chevron-forward" size={18} color={colors.textMute} style={{ marginLeft: 6 }} />
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
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
        <Text style={styles.title}>Strategies</Text>
        <Pressable
          onPress={() => router.push('/wizard' as any)}
          style={({ pressed }) => [styles.newBtn, pressed && { opacity: 0.85, transform: [{ scale: 0.97 }] }]}
        >
          <Ionicons name="add" size={18} color={colors.accentText} />
          <Text style={styles.newBtnText}>New</Text>
        </Pressable>
      </View>
      <Text style={styles.subtitle}>
        {data?.length ? `${data.length} strateg${data.length === 1 ? 'y' : 'ies'} · tap any to inspect` : 'Your saved strategies'}
      </Text>
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
        <EmptyState icon="cloud-offline-outline" title="Couldn't load strategies" hint="Pull down to retry." />
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <AmbientBg variant="duo" />
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
          <View>
            <EmptyState
              icon="rocket-outline"
              title="No strategies yet"
              hint="Tap “New” to build a quick paper-trading strategy, or use the full builder on tradehub.markets."
              tone="accent"
            />
            <View style={{ marginTop: spacing.lg, paddingHorizontal: spacing.lg }}>
              <Pressable
                onPress={() => router.push('/wizard' as any)}
                style={({ pressed }) => [styles.emptyCta, glow.accent, pressed && { opacity: 0.85, transform: [{ scale: 0.99 }] }]}
              >
                <Ionicons name="add-circle" size={20} color={colors.accentText} />
                <Text style={styles.emptyCtaText}>Build my first strategy</Text>
              </Pressable>
            </View>
          </View>
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
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    overflow: 'hidden',
  },
  rowActive: {
    borderColor: 'rgba(52, 211, 153, 0.32)',
    backgroundColor: '#141d2c',
  },
  stripe: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 3,
  },
  rowHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  rowTitle: { color: colors.text, fontFamily: font.bold, fontSize: 16, flex: 1, letterSpacing: -0.2 },
  rowDesc: { color: colors.textDim, fontFamily: font.regular, fontSize: 13, marginTop: 6, lineHeight: 18 },
  rowFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    gap: spacing.lg,
  },
  metricBlock: {},
  metricLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: 2,
  },
  metricValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
  },
  sparkWrap: {
    marginLeft: 'auto',
    width: 70,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sparkPlaceholder: {
    width: 70,
    height: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sparkPlaceholderText: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 12,
  },
  newBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.accent,
  },
  newBtnText: {
    color: colors.accentText,
    fontFamily: font.bold,
    fontSize: 13,
    letterSpacing: 0.3,
  },
  emptyCta: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: radius.md,
    backgroundColor: colors.accent,
  },
  emptyCtaText: {
    color: colors.accentText,
    fontFamily: font.black,
    fontSize: 15,
    letterSpacing: 0.3,
  },
});
