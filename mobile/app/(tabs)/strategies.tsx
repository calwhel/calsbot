import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Pressable,
  ActivityIndicator,
  RefreshControl,
  ScrollView,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, RadialGradient, Stop, Rect } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { StrategyListSkeleton } from '@/components/Skeleton';
import { Pill } from '@/components/Pill';
import { Sparkline } from '@/components/Sparkline';
import { AmbientBg } from '@/components/AmbientBg';
import { CoinChip } from '@/components/CoinChip';
import { MiniDonut } from '@/components/MiniDonut';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type Strategy } from '@/lib/api';

type StatusFilter = 'all' | 'active' | 'paused' | 'archived';

const FILTERS: Array<{ key: StatusFilter; label: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { key: 'all',      label: 'All',      icon: 'apps-outline' },
  { key: 'active',   label: 'Active',   icon: 'flash' },
  { key: 'paused',   label: 'Paused',   icon: 'pause' },
  { key: 'archived', label: 'Archived', icon: 'archive-outline' },
];

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

function fmtPips(v: number): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(1)} pips`;
}

/** Synthetic sparkline trajectory (no per-trade equity available). */
function buildSpark(pnl: number, wr: number, trades: number): number[] {
  if (trades < 2) return [];
  const n = Math.min(Math.max(8, trades), 20);
  const winBias = (wr - 50) / 50;
  const out: number[] = [];
  let cum = 0;
  for (let i = 0; i < n; i++) {
    const noise = Math.sin(i * 1.7 + pnl) * 0.4;
    cum += winBias + noise;
    out.push(cum);
  }
  if (pnl !== 0 && out[out.length - 1] !== 0) {
    const scale = pnl / out[out.length - 1];
    return out.map((v) => v * Math.abs(scale));
  }
  return out;
}

const MARKET_LABEL: Record<string, string> = {
  crypto: '₿ Crypto',
  forex:  '💱 Forex',
  stock:  '📈 Stocks',
  index:  '📊 Indices',
};
const MARKET_COLOR: Record<string, string> = {
  crypto: '#5B6CF7',
  forex:  '#22c55e',
  stock:  '#f59e0b',
  index:  '#a855f7',
};

const StrategyCard = React.memo(function StrategyCard({ s, onPress }: { s: Strategy; onPress: () => void }) {
  const perf = s.performance || {};
  const pnl = perf.total_pnl ?? 0;
  const pipsPnl = perf.total_pips_pnl ?? null;
  const trades = perf.total_trades ?? 0;
  const wr = perf.win_rate ?? 0;
  const spark = buildSpark(pnl, wr, trades);
  const isActive = s.status === 'active';
  const symbol = (s.config?.symbol as string) || 'BTC';
  const timeframe = (s.config?.timeframe as string) || '';
  const _cfg = s.config as Record<string, any> | undefined;
  const assetClass = (_cfg?._asset_class || _cfg?.asset_class || 'crypto') as string;
  const isForexLike = ['forex', 'index', 'metals', 'commodity'].includes(assetClass);
  const marketLabel = MARKET_LABEL[assetClass] ?? assetClass.toUpperCase();
  const marketColor = MARKET_COLOR[assetClass] ?? '#5B6CF7';

  const uid = React.useId().replace(/:/g, '');
  const haloId = `card-halo-${uid}`;
  const sheenId = `card-sheen-${uid}`;
  const displayPnl = isForexLike ? (pipsPnl ?? 0) : pnl;
  const tonePrimary = displayPnl > 0 ? colors.positive : displayPnl < 0 ? colors.negative : colors.textDim;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        isActive && styles.cardActive,
        isActive && glow.positive,
        !isActive && glow.card,
        pressed && { opacity: 0.92, transform: [{ scale: 0.995 }] },
      ]}
    >
      {/* Background washes */}
      <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <SvgLinearGradient id={sheenId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor={colors.card} stopOpacity="1" />
            <Stop offset="1" stopColor={colors.card} stopOpacity="1" />
          </SvgLinearGradient>
          <RadialGradient id={haloId} cx="100%" cy="0%" rx="70%" ry="60%">
            <Stop offset="0" stopColor={tonePrimary} stopOpacity={trades > 0 ? '0.18' : '0'} />
            <Stop offset="1" stopColor={tonePrimary} stopOpacity="0" />
          </RadialGradient>
        </Defs>
        <Rect width="100" height="100" fill={`url(#${sheenId})`} />
        <Rect width="100" height="100" fill={`url(#${haloId})`} />
      </Svg>

      {/* Active stripe */}
      <View style={[styles.stripe, { backgroundColor: isActive ? colors.positive : 'transparent' }]} />

      <View style={styles.cardInner}>
        {/* Top row: coin + name + status */}
        <View style={styles.topRow}>
          <CoinChip symbol={symbol} size={42} />
          <View style={{ flex: 1, marginLeft: spacing.md }}>
            <Text style={styles.title} numberOfLines={1}>{s.name || 'Unnamed Strategy'}</Text>
            <View style={styles.subRow}>
              <View style={[styles.mktBadge, { backgroundColor: `${marketColor}22`, borderColor: `${marketColor}55` }]}>
                <Text style={[styles.mktBadgeText, { color: marketColor }]}>{marketLabel}</Text>
              </View>
              <View style={styles.dotSep} />
              <Text style={styles.subText} numberOfLines={1}>
                {symbol}{timeframe ? ` · ${timeframe}` : ''}
              </Text>
              {s.is_locked ? (
                <>
                  <View style={styles.dotSep} />
                  <Ionicons name="lock-closed" size={11} color={colors.warning} />
                  <Text style={[styles.subText, { color: colors.warning }]}>Locked</Text>
                </>
              ) : null}
            </View>
          </View>
          <Pill label={s.status} tone={statusTone(s.status)} small />
        </View>

        {/* Description */}
        {s.description ? (
          <Text style={styles.desc} numberOfLines={2}>{s.description}</Text>
        ) : null}

        {/* Metrics row */}
        <View style={styles.metricsRow}>
          <View style={styles.metricMain}>
            <Text style={styles.metricLabel}>{isForexLike ? 'Pips' : 'P&L'}</Text>
            <Text
              style={[
                styles.metricMainValue,
                { color: displayPnl > 0 ? colors.positive : displayPnl < 0 ? colors.negative : colors.text },
              ]}
            >
              {trades > 0 ? (isForexLike ? fmtPips(pipsPnl ?? 0) : fmtPnl(pnl)) : '—'}
            </Text>
          </View>

          <View style={styles.metricsSep} />

          <View style={styles.metricCol}>
            <Text style={styles.metricLabel}>Trades</Text>
            <Text style={styles.metricColValue}>{trades}</Text>
          </View>

          <View style={styles.metricsSep} />

          <View style={[styles.metricCol, { alignItems: 'flex-end' }]}>
            <Text style={styles.metricLabel}>Win rate</Text>
            {trades > 0 ? (
              <View style={styles.donutRow}>
                <Text style={styles.metricColValue}>{wr.toFixed(0)}%</Text>
                <MiniDonut value={wr} size={26} stroke={3.5} label=" " />
              </View>
            ) : (
              <Text style={styles.metricColValue}>—</Text>
            )}
          </View>
        </View>

        {/* Sparkline strip */}
        <View style={styles.sparkRow}>
          {spark.length >= 2 ? (
            <Sparkline values={spark} width={280} height={36} strokeWidth={1.8} />
          ) : (
            <View style={styles.sparkPlaceholder}>
              <Text style={styles.sparkPlaceholderText}>No trade history yet — activate to start collecting fills.</Text>
            </View>
          )}
          <View style={styles.chevWrap}>
            <Ionicons name="chevron-forward" size={16} color={colors.textMute} />
          </View>
        </View>
      </View>
    </Pressable>
  );
});

function FilterChips({
  value,
  counts,
  onChange,
}: {
  value: StatusFilter;
  counts: Record<StatusFilter, number>;
  onChange: (f: StatusFilter) => void;
}) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.chipRow}
    >
      {FILTERS.map((f) => {
        const active = f.key === value;
        return (
          <Pressable
            key={f.key}
            onPress={() => {
              if (active) return;
              Haptics.selectionAsync().catch(() => {});
              onChange(f.key);
            }}
            style={({ pressed }) => [
              styles.chip,
              active && styles.chipActive,
              pressed && { opacity: 0.85 },
            ]}
          >
            <Ionicons
              name={f.icon}
              size={13}
              color={active ? colors.accentText : colors.textDim}
            />
            <Text style={[styles.chipText, active && styles.chipTextActive]}>{f.label}</Text>
            <View style={[styles.chipCount, active && styles.chipCountActive]}>
              <Text style={[styles.chipCountText, active && styles.chipCountTextActive]}>
                {counts[f.key]}
              </Text>
            </View>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

export default function StrategiesScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const [filter, setFilter] = useState<StatusFilter>('all');

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['strategies', uid],
    queryFn: () => apiGet<Strategy[]>('/api/strategies', uid),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  const counts = useMemo(() => {
    const list = data || [];
    return {
      all:      list.length,
      active:   list.filter((s) => s.status === 'active').length,
      paused:   list.filter((s) => s.status === 'paused').length,
      archived: list.filter((s) => s.status === 'archived').length,
    } as Record<StatusFilter, number>;
  }, [data]);

  const filtered = useMemo(() => {
    const list = data || [];
    if (filter === 'all') return list;
    return list.filter((s) => s.status === filter);
  }, [data, filter]);

  const renderItem = useCallback(
    ({ item }: { item: Strategy }) => (
      <StrategyCard s={item} onPress={() => router.push(`/strategy/${item.id}`)} />
    ),
    [router],
  );

  const Header = (
    <View style={styles.header}>
      <View style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }}>
        <View style={{ flex: 1 }}>
          <Text style={styles.titleHero}>Strategies</Text>
          <Text style={styles.subtitleHero}>
            {data?.length
              ? `${data.length} saved · ${counts.active} active right now`
              : 'Build, backtest, and run trading strategies'}
          </Text>
        </View>
        <Pressable
          onPress={() => router.push('/build' as any)}
          style={({ pressed }) => [styles.newBtn, pressed && { opacity: 0.85, transform: [{ scale: 0.96 }] }]}
        >
          <Ionicons name="add" size={18} color={colors.accentText} />
          <Text style={styles.newBtnText}>New</Text>
        </Pressable>
      </View>

      <FilterChips value={filter} counts={counts} onChange={setFilter} />

      {/* Forex scanner shortcut */}
      <Pressable
        onPress={() => {
          Haptics.selectionAsync();
          router.push('/forex-scanner' as any);
        }}
        style={({ pressed }) => [styles.scannerBanner, pressed && { opacity: 0.8 }]}
      >
        <Ionicons name="radio-outline" size={14} color="#639BEB" />
        <Text style={styles.scannerBannerText}>Forex Scanner</Text>
        <Text style={styles.scannerBannerSub}>BOS · CHoCH · FVG · OB across 11 pairs</Text>
        <Ionicons name="chevron-forward" size={13} color={colors.textMute} style={{ marginLeft: 'auto' }} />
      </Pressable>

      {/* Portfolio review shortcut */}
      <Pressable
        onPress={() => {
          Haptics.selectionAsync();
          router.push('/portfolio' as any);
        }}
        style={({ pressed }) => [styles.scannerBanner, pressed && { opacity: 0.8 }]}
      >
        <Ionicons name="bar-chart-outline" size={14} color="#d4a017" />
        <Text style={styles.scannerBannerText}>Portfolio Review</Text>
        <Text style={styles.scannerBannerSub}>AI review of your whole strategy book</Text>
        <Ionicons name="chevron-forward" size={13} color={colors.textMute} style={{ marginLeft: 'auto' }} />
      </Pressable>
    </View>
  );

  if (isLoading) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <AmbientBg variant="duo" />
        {Header}
        <View style={{ paddingHorizontal: spacing.lg, paddingTop: spacing.md }}>
          <StrategyListSkeleton count={4} />
        </View>
      </View>
    );
  }

  if (isError) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <AmbientBg variant="duo" />
        <ScrollView
          contentContainerStyle={{ flexGrow: 1 }}
          refreshControl={
            <RefreshControl
              refreshing={isFetching && !isLoading}
              onRefresh={onRefresh}
              tintColor={colors.accent}
              colors={[colors.accent]}
              progressBackgroundColor={colors.bgElev}
            />
          }
        >
          {Header}
          <EmptyState icon="cloud-offline-outline" title="Couldn't load strategies" hint="Pull down to retry." ctaLabel="Retry" onCta={() => refetch()} />
        </ScrollView>
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <AmbientBg variant="duo" />
      <FlatList
        data={filtered}
        keyExtractor={(s) => `s-${s.id}`}
        renderItem={renderItem}
        ListHeaderComponent={Header}
        ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
        contentContainerStyle={styles.listContent}
        initialNumToRender={6}
        maxToRenderPerBatch={4}
        windowSize={7}
        removeClippedSubviews
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
          filter === 'all' ? (
            <View>
              <EmptyState
                icon="rocket-outline"
                title="No strategies yet"
                hint="Tap “New” to build a quick paper-trading strategy, or use the full builder on tradehub.markets."
                tone="accent"
              />
              <View style={{ marginTop: spacing.lg, paddingHorizontal: spacing.lg }}>
                <Pressable
                  onPress={() => router.push('/build' as any)}
                  style={({ pressed }) => [styles.emptyCta, glow.accent, pressed && { opacity: 0.85, transform: [{ scale: 0.99 }] }]}
                >
                  <Ionicons name="add-circle" size={20} color={colors.accentText} />
                  <Text style={styles.emptyCtaText}>Build my first strategy</Text>
                </Pressable>
              </View>
            </View>
          ) : (
            <EmptyState
              icon="filter-outline"
              title={`No ${filter} strategies`}
              hint="Try a different filter — your other strategies are still saved."
            />
          )
        }
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.sm },
  titleHero: { color: colors.text, fontFamily: font.black, fontSize: 32, letterSpacing: -1.0 },
  subtitleHero: { color: colors.textDim, fontFamily: font.regular, fontSize: 14, marginTop: 4, lineHeight: 19 },
  listContent: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl + 96 },

  // Card
  card: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    overflow: 'hidden',
    minHeight: 168,
  },
  cardActive: {
    borderColor: 'rgba(52,211,153,0.38)',
  },
  cardInner: {
    padding: spacing.lg,
  },
  stripe: {
    position: 'absolute',
    left: 0,
    top: 0,
    bottom: 0,
    width: 3,
  },
  topRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  title: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 17,
    letterSpacing: -0.4,
  },
  subRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 3,
  },
  subText: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 12,
  },
  dotSep: {
    width: 3,
    height: 3,
    borderRadius: 1.5,
    backgroundColor: colors.textMute,
  },
  mktBadge: {
    borderWidth: 1,
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 1,
  },
  mktBadgeText: {
    fontSize: 10,
    fontFamily: font.semibold,
  },
  desc: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 18,
    marginTop: spacing.md,
  },

  metricsRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    marginTop: spacing.lg,
    gap: spacing.md,
  },
  metricMain: { flex: 1.2 },
  metricCol: { flex: 1 },
  metricsSep: {
    width: 1,
    height: 32,
    backgroundColor: colors.divider,
  },
  metricLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  metricMainValue: {
    fontFamily: font.black,
    fontSize: 24,
    letterSpacing: -0.8,
    fontVariant: ['tabular-nums'],
  },
  metricColValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    letterSpacing: -0.2,
    fontVariant: ['tabular-nums'],
  },
  donutRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },

  sparkRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  sparkPlaceholder: {
    flex: 1,
    paddingVertical: 8,
  },
  sparkPlaceholderText: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11.5,
  },
  chevWrap: {
    marginLeft: 'auto',
    paddingLeft: 8,
  },

  // Chips
  chipRow: {
    paddingTop: spacing.lg,
    paddingBottom: 4,
    gap: 8,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: colors.cardHi,
    borderColor: colors.border,
    borderWidth: 1,
    paddingHorizontal: 14,
    paddingVertical: 11,
    minHeight: 44,
    borderRadius: radius.pill,
  },
  chipActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  chipText: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.3,
  },
  chipTextActive: { color: colors.accentText },
  chipCount: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    minWidth: 18,
    height: 18,
    borderRadius: 9,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 4,
  },
  chipCountActive: { backgroundColor: 'rgba(0,0,0,0.22)' },
  chipCountText: {
    color: colors.textDim,
    fontFamily: font.black,
    fontSize: 10,
    letterSpacing: 0.2,
  },
  chipCountTextActive: { color: colors.accentText },

  // CTAs
  newBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: radius.pill,
    backgroundColor: colors.accent,
    ...glow.accent,
  },
  newBtnText: {
    color: colors.accentText,
    fontFamily: font.bold,
    fontSize: 13,
    letterSpacing: 0.3,
  },
  scannerBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: spacing.sm,
    paddingHorizontal: 12,
    paddingVertical: 10,
    borderRadius: radius.md,
    backgroundColor: colors.card,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderHi,
  },
  scannerBannerText: {
    fontFamily: font.bold,
    fontSize: 13,
    color: '#639BEB',
  },
  scannerBannerSub: {
    fontFamily: font.regular,
    fontSize: 11,
    color: colors.textMute,
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
