import React, { useCallback, useMemo } from 'react';
import {
  View, Text, StyleSheet, ScrollView, ActivityIndicator,
  RefreshControl, useWindowDimensions, Alert,
} from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { StatCard } from '@/components/StatCard';
import { EquityCurve } from '@/components/EquityCurve';
import { CandleChart, type Candle, type ChartMarker } from '@/components/CandleChart';
import { PrimaryButton } from '@/components/PrimaryButton';
import { StrategyConfigCard } from '@/components/StrategyConfigCard';
import { AutomationCard } from '@/components/AutomationCard';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPost, type Strategy } from '@/lib/api';

type TradeRow = {
  id?: number;
  symbol: string;
  direction: string;
  outcome: string;
  pnl_pct: number | null;
  fired_at: string | null;
  closed_at?: string | null;
  entry_price?: number | null;
  exit_price?: number | null;
};

type TradesResponse = {
  trades: TradeRow[];
  total?: number;
};

type ToggleResponse = { status: string };

function fmtPnl(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function shortDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return `${(d.getMonth() + 1).toString().padStart(2, '0')}/${d.getDate().toString().padStart(2, '0')} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
  } catch { return '—'; }
}

function statusTone(status: string): 'positive' | 'warning' | 'neutral' | 'negative' {
  if (status === 'active') return 'positive';
  if (status === 'paused') return 'warning';
  if (status === 'archived') return 'negative';
  return 'neutral';
}

export default function StrategyDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const sid = Number(id);
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const { width } = useWindowDimensions();
  const chartW = width - spacing.lg * 2 - spacing.sm * 2;
  const qc = useQueryClient();

  // We piggyback on the existing /api/strategies list (which already returns
  // perf + recent trades for this user) instead of using the heavier detail
  // endpoint, which is wizard-config oriented.
  const listQ = useQuery({
    queryKey: ['strategies', uid],
    queryFn: () => apiGet<Strategy[]>('/api/strategies', uid),
    enabled: !!uid,
  });

  const strategy = useMemo(
    () => (listQ.data || []).find((s) => s.id === sid),
    [listQ.data, sid],
  );

  const tradesQ = useQuery({
    queryKey: ['strategy-trades', sid, uid],
    queryFn: () => apiGet<TradesResponse>(`/api/strategies/${sid}/trades`, uid, { limit: 30 }),
    enabled: !!uid && !!sid,
  });

  // Pick the symbol to chart: most-recent traded symbol on this strategy. If
  // the strategy has never fired, fall back to (a) the first symbol in its
  // configured universe, or (b) BTCUSDT as a sensible default — so new users
  // see actual market context instead of a blank section.
  const chartSymbol = useMemo<string>(() => {
    const trades = tradesQ.data?.trades || [];
    if (trades.length > 0) return trades[0].symbol;

    const cfg = (strategy?.config || {}) as Record<string, unknown>;
    const universe = (cfg.universe || {}) as Record<string, unknown>;
    if (universe.type === 'specific' && Array.isArray(universe.symbols) && universe.symbols.length > 0) {
      const first = universe.symbols[0];
      if (typeof first === 'string' && first.length > 0) return first;
    }
    const legacySymbols = (cfg as { symbols?: unknown }).symbols;
    if (Array.isArray(legacySymbols) && legacySymbols.length > 0 && typeof legacySymbols[0] === 'string') {
      return legacySymbols[0] as string;
    }
    return 'BTCUSDT';
  }, [tradesQ.data, strategy?.config]);

  const hasFiredHere = (tradesQ.data?.trades || []).length > 0;

  const candlesQ = useQuery({
    queryKey: ['candles', chartSymbol],
    queryFn: () => apiGet<{ candles: Candle[]; symbol: string; tf: string }>(
      `/api/candles/${chartSymbol!.replace('USDT', '')}`,
      uid,
      { tf: '5m', limit: 80 },
    ),
    enabled: !!uid && !!chartSymbol,
    staleTime: 60_000,
    refetchInterval: 60_000, // refresh once per minute
  });

  const toggleM = useMutation({
    mutationFn: () => apiPost<ToggleResponse>(`/api/strategies/${sid}/toggle`, {}, uid),
    // Optimistic flip — feels instant. The server is the source of truth, so
    // we still invalidate on success to pick up any side-effects (health
    // score, status normalisation), and rollback to the snapshot on error.
    onMutate: async () => {
      await qc.cancelQueries({ queryKey: ['strategies', uid] });
      const prev = qc.getQueryData<Strategy[]>(['strategies', uid]);
      if (prev) {
        qc.setQueryData<Strategy[]>(
          ['strategies', uid],
          prev.map((s) =>
            s.id === sid
              ? { ...s, status: s.status === 'active' ? 'paused' : 'active' }
              : s,
          ),
        );
      }
      return { prev };
    },
    onError: (e, _vars, ctx) => {
      if (ctx?.prev) qc.setQueryData(['strategies', uid], ctx.prev);
      Alert.alert('Could not toggle strategy', (e as Error).message || 'Try again.');
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: ['strategies', uid] });
      qc.invalidateQueries({ queryKey: ['portfolio', uid] });
    },
  });

  // Build entry/exit markers for trades whose symbol matches the charted one.
  const chartMarkers = useMemo<ChartMarker[]>(() => {
    if (!chartSymbol || !candlesQ.data?.candles?.length) return [];
    const trades = tradesQ.data?.trades || [];
    const tMin = candlesQ.data.candles[0].time;
    const tMax = candlesQ.data.candles[candlesQ.data.candles.length - 1].time;
    const out: ChartMarker[] = [];
    for (const t of trades) {
      if (t.symbol !== chartSymbol) continue;
      // entry
      if (t.fired_at && t.entry_price != null) {
        const ts = Math.floor(new Date(t.fired_at).getTime() / 1000);
        if (ts >= tMin && ts <= tMax) {
          out.push({
            time: ts,
            price: t.entry_price,
            kind: 'open',
            direction: t.direction === 'SHORT' ? 'SHORT' : 'LONG',
          });
        }
      }
      // exit (only for closed trades within the chart window)
      if (t.closed_at && t.exit_price != null && t.outcome !== 'OPEN') {
        const ts = Math.floor(new Date(t.closed_at).getTime() / 1000);
        if (ts >= tMin && ts <= tMax) {
          out.push({
            time: ts,
            price: t.exit_price,
            kind: t.outcome === 'WIN' ? 'close-win' : 'close-loss',
          });
        }
      }
    }
    return out;
  }, [chartSymbol, candlesQ.data, tradesQ.data]);

  const equityValues = useMemo(() => {
    const trades = tradesQ.data?.trades || [];
    // trades come most-recent-first; reverse to chronological for cumulative curve
    const closed = [...trades]
      .filter((t) => t.outcome && t.outcome !== 'OPEN' && t.pnl_pct !== null)
      .reverse();
    let cum = 0;
    return closed.map((t) => {
      cum += t.pnl_pct as number;
      return Math.round(cum * 100) / 100;
    });
  }, [tradesQ.data]);

  const onRefresh = useCallback(() => {
    listQ.refetch();
    tradesQ.refetch();
  }, [listQ, tradesQ]);

  // Show a spinner whenever we don't yet have a definitive answer about whether
  // this strategy exists — prevents a "not found" flash on deep link / reload
  // when the cached strategies list is still being (re)hydrated.
  if (listQ.isLoading || (!listQ.data && listQ.isFetching)) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <View style={[styles.center, { backgroundColor: colors.bg, flex: 1 }]}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      </>
    );
  }

  if (!strategy) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <EmptyState icon="alert-circle-outline" title="Strategy not found" hint="It may have been archived or removed." />
      </>
    );
  }

  const perf = strategy.performance || {};
  const pnl = perf.total_pnl ?? 0;
  const wr = perf.win_rate ?? 0;
  const trades = perf.total_trades ?? 0;
  const trades_list = tradesQ.data?.trades || [];
  const isActive = strategy.status === 'active';

  return (
    <>
      <Stack.Screen options={{ title: '' }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: colors.bg }}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 32 }]}
        refreshControl={
          <RefreshControl
            refreshing={(listQ.isFetching || tradesQ.isFetching) && !listQ.isLoading}
            onRefresh={onRefresh}
            tintColor={colors.accent}
            colors={[colors.accent]}
            progressBackgroundColor={colors.bgElev}
          />
        }
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title} numberOfLines={2}>{strategy.name}</Text>
          <View style={styles.headerMeta}>
            <Pill label={strategy.status} tone={statusTone(strategy.status)} small />
            {strategy.is_locked ? <Pill label="Locked" tone="warning" small /> : null}
            {strategy.is_public ? <Pill label="Public" tone="accent" small /> : null}
          </View>
          {strategy.description ? (
            <Text style={styles.desc}>{strategy.description}</Text>
          ) : null}
        </View>

        {/* Toggle */}
        {!strategy.is_locked && (
          <View style={{ marginTop: spacing.lg }}>
            <PrimaryButton
              label={isActive ? 'Pause this strategy' : 'Activate this strategy'}
              variant={isActive ? 'secondary' : 'primary'}
              onPress={() => toggleM.mutate()}
              loading={toggleM.isPending}
            />
            {!isActive ? (
              <Text style={styles.activateHint}>
                While paused, conditions are not checked and no new trades will fire.
              </Text>
            ) : null}
          </View>
        )}

        {/* How this strategy works — parsed config */}
        <View style={{ marginTop: spacing.lg }}>
          <StrategyConfigCard config={strategy.config} />
        </View>

        {/* Run backtest CTA — secondary because Activate is the primary action */}
        <View style={{ marginTop: spacing.lg }}>
          <PrimaryButton
            label="Run backtest on this strategy"
            variant="secondary"
            onPress={() => router.push(`/backtest/${sid}`)}
            icon={<Ionicons name="time" size={16} color={colors.text} />}
          />
          <Text style={styles.activateHint}>
            See how this strategy would have performed over the last 30 or 90 days.
          </Text>
        </View>

        {/* Stats */}
        <View style={[styles.statRow, { marginTop: spacing.lg }]}>
          <StatCard
            label="Total P&L"
            value={trades > 0 ? fmtPnl(pnl) : '—'}
            tone={pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral'}
          />
          <View style={{ width: spacing.md }} />
          <StatCard label="Win rate" value={trades > 0 ? `${wr.toFixed(1)}%` : '—'} sub={`${trades} closed`} />
        </View>
        <View style={[styles.statRow, { marginTop: spacing.md }]}>
          <StatCard
            label="Best trade"
            value={trades > 0 ? fmtPnl(perf.best_trade ?? 0) : '—'}
            tone="positive"
            compact
          />
          <View style={{ width: spacing.md }} />
          <StatCard
            label="Worst trade"
            value={trades > 0 ? fmtPnl(perf.worst_trade ?? 0) : '—'}
            tone="negative"
            compact
          />
          <View style={{ width: spacing.md }} />
          <StatCard
            label="Open"
            value={`${perf.open_trades ?? 0}`}
            sub={(perf.open_trades ?? 0) > 0 ? 'live' : ''}
            compact
          />
        </View>

        {/* Price action (candles + entry/exit markers if any). Always shown —
            even before the strategy has fired — so users get market context. */}
        <View style={{ marginTop: spacing.lg }}>
          <View style={styles.sectionHead}>
            <Text style={styles.sectionLabel}>Live price · {chartSymbol.replace('USDT', '')}</Text>
            {!hasFiredHere ? (
              <Text style={styles.sectionHint}>preview</Text>
            ) : null}
          </View>
          {candlesQ.isLoading ? (
            <View style={[styles.card, styles.center, { paddingVertical: spacing.xl }]}>
              <ActivityIndicator color={colors.accent} />
            </View>
          ) : (
            <CandleChart
              candles={candlesQ.data?.candles || []}
              markers={chartMarkers}
              width={chartW}
              height={180}
              symbol={chartSymbol}
              tf="5m"
            />
          )}
          {!hasFiredHere ? (
            <Text style={styles.previewNote}>
              Showing recent {chartSymbol.replace('USDT', '')} candles. Once this strategy fires, entry and exit markers will plot here automatically.
            </Text>
          ) : null}
        </View>

        {/* Equity curve */}
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.sectionLabel}>Equity curve</Text>
          <EquityCurve values={equityValues} width={chartW} height={160} />
        </View>

        {/* How automation works — educational explainer */}
        <View style={{ marginTop: spacing.lg }}>
          <AutomationCard />
        </View>

        {/* Recent trades */}
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.sectionLabel}>Recent trades</Text>
          {tradesQ.isLoading ? (
            <View style={[styles.card, styles.center, { paddingVertical: spacing.xl }]}>
              <ActivityIndicator color={colors.accent} />
            </View>
          ) : trades_list.length === 0 ? (
            <EmptyState
              icon="hourglass-outline"
              title="No trades yet"
              hint={isActive ? "This strategy is live and watching the market — when conditions match, the first trade will appear here." : 'Activate this strategy above to start collecting trades.'}
            />
          ) : (
            <View style={styles.tradeList}>
              {trades_list.slice(0, 30).map((t, i) => (
                <View key={`t-${i}`} style={[styles.tradeRow, i < trades_list.length - 1 && styles.tradeRowDiv]}>
                  <View style={styles.tradeLeft}>
                    <Text style={styles.tradeSymbol}>{t.symbol}</Text>
                    <View style={styles.tradeMeta}>
                      <Pill
                        label={t.direction}
                        tone={t.direction === 'LONG' ? 'positive' : 'negative'}
                        small
                      />
                      <Text style={styles.tradeTime}>{shortDate(t.fired_at)}</Text>
                    </View>
                  </View>
                  <View style={styles.tradeRight}>
                    <Text style={[
                      styles.tradePnl,
                      {
                        color:
                          t.outcome === 'WIN' ? colors.positive :
                          t.outcome === 'LOSS' ? colors.negative :
                          t.outcome === 'OPEN' ? colors.accent :
                          colors.textDim,
                      },
                    ]}>
                      {t.outcome === 'OPEN' ? 'OPEN' : fmtPnl(t.pnl_pct)}
                    </Text>
                    <Text style={styles.tradeOutcome}>{t.outcome}</Text>
                  </View>
                </View>
              ))}
            </View>
          )}
        </View>
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.sm, paddingBottom: spacing.xxl + 96 },
  center: { alignItems: 'center', justifyContent: 'center', flex: 1, padding: spacing.xl },
  header: { paddingTop: spacing.sm },
  title: { color: colors.text, fontFamily: font.black, fontSize: 26, letterSpacing: -0.6 },
  headerMeta: { flexDirection: 'row', gap: 6, marginTop: spacing.sm, flexWrap: 'wrap' },
  desc: { color: colors.textDim, fontFamily: font.regular, fontSize: 14, marginTop: spacing.md, lineHeight: 20 },
  activateHint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  statRow: { flexDirection: 'row' },
  sectionHead: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.sm,
  },
  sectionLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
  sectionHint: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: radius.pill,
    marginBottom: spacing.sm,
  },
  previewNote: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11.5,
    lineHeight: 16,
    marginTop: spacing.sm,
    paddingHorizontal: spacing.xs,
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tradeList: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  tradeRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: spacing.lg, paddingVertical: 12,
  },
  tradeRowDiv: { borderBottomWidth: 1, borderBottomColor: colors.divider },
  tradeLeft: { flex: 1 },
  tradeRight: { alignItems: 'flex-end' },
  tradeSymbol: { color: colors.text, fontFamily: font.bold, fontSize: 15 },
  tradeMeta: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 },
  tradeTime: { color: colors.textMute, fontFamily: font.regular, fontSize: 11 },
  tradePnl: { fontFamily: font.bold, fontSize: 15, fontVariant: ['tabular-nums'] },
  tradeOutcome: {
    color: colors.textMute, fontFamily: font.bold, fontSize: 10,
    letterSpacing: 0.5, marginTop: 2,
  },
});
