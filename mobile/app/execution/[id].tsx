import React, { useMemo } from 'react';
import {
  View, Text, StyleSheet, ScrollView, ActivityIndicator,
  RefreshControl, useWindowDimensions, Pressable,
} from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { CandleChart, type ChartMarker, type ChartPriceLine } from '@/components/CandleChart';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type ExecutionDetail } from '@/lib/api';

function fmtPnl(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}
function fmtUsd(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : v < 0 ? '−' : '';
  return `${sign}$${Math.abs(v).toFixed(2)}`;
}
function fmtPx(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '—';
  if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (p >= 1)    return p.toFixed(3);
  if (p >= 0.01) return p.toFixed(5);
  return p.toFixed(8);
}
function shortDate(iso: string | null): string {
  if (!iso) return '—';
  try {
    const d = new Date(iso);
    return `${(d.getMonth() + 1).toString().padStart(2, '0')}/${d.getDate().toString().padStart(2, '0')} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
  } catch { return '—'; }
}
function outcomeTone(o: string): { fg: string; bg: string; label: string } {
  if (o === 'WIN')       return { fg: colors.positive, bg: colors.positiveDim, label: 'WIN' };
  if (o === 'LOSS')      return { fg: colors.negative, bg: colors.negativeDim, label: 'LOSS' };
  if (o === 'BREAKEVEN') return { fg: colors.textDim,  bg: colors.bgElev,      label: 'BREAKEVEN' };
  return { fg: colors.accent, bg: colors.accentDim, label: 'OPEN' };
}

export default function ExecutionDetailScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const eid = Number(id);
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const { width } = useWindowDimensions();
  const chartW = width - spacing.lg * 2 - spacing.sm * 2;

  const { data, isLoading, isError, refetch, isFetching, error } = useQuery({
    queryKey: ['execution', uid, eid],
    queryFn: () => apiGet<ExecutionDetail>(`/api/executions/${eid}`, uid),
    enabled: !!uid && Number.isFinite(eid),
  });

  const candles = data?.candles?.data || [];
  const tone = outcomeTone(data?.outcome || 'OPEN');

  // Place entry marker at fired_at (nearest candle handled by CandleChart),
  // and exit marker at closed_at if the trade is closed.
  const markers = useMemo<ChartMarker[]>(() => {
    if (!data) return [];
    const out: ChartMarker[] = [];
    if (data.fired_at && data.entry_price != null) {
      out.push({
        time: Math.floor(new Date(data.fired_at).getTime() / 1000),
        price: data.entry_price,
        kind: 'open',
        direction: data.direction === 'SHORT' ? 'SHORT' : 'LONG',
      });
    }
    if (data.closed_at && data.exit_price != null && data.outcome !== 'OPEN') {
      out.push({
        time: Math.floor(new Date(data.closed_at).getTime() / 1000),
        price: data.exit_price,
        kind: data.outcome === 'WIN' ? 'close-win' : 'close-loss',
      });
    }
    return out;
  }, [data]);

  // TP / SL / TP2 as horizontal price lines so they read as plan vs reality.
  const priceLines = useMemo<ChartPriceLine[]>(() => {
    if (!data) return [];
    const lines: ChartPriceLine[] = [];
    const isLong = data.direction !== 'SHORT';
    if (data.entry_price != null) {
      lines.push({ price: data.entry_price, side: isLong ? 'buy' : 'sell', label: 'ENTRY' });
    }
    if (data.tp_price != null) {
      lines.push({ price: data.tp_price, side: isLong ? 'buy' : 'sell', label: 'TP' });
    }
    if (data.tp2_price != null) {
      lines.push({ price: data.tp2_price, side: isLong ? 'buy' : 'sell', label: 'TP2' });
    }
    if (data.sl_price != null) {
      lines.push({ price: data.sl_price, side: isLong ? 'sell' : 'buy', label: 'SL' });
    }
    return lines;
  }, [data]);

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <Stack.Screen options={{ headerShown: false }} />

      <View style={styles.topbar}>
        <Pressable onPress={() => router.back()} hitSlop={12} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={22} color={colors.text} />
        </Pressable>
        <Text style={styles.topbarTitle} numberOfLines={1}>
          {data ? `${data.symbol.replace('USDT', '')} · ${data.direction}` : 'Trade'}
        </Text>
        <View style={{ width: 32 }} />
      </View>

      {isLoading ? (
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      ) : isError || !data ? (
        <ScrollView
          contentContainerStyle={{ flexGrow: 1 }}
          refreshControl={
            <RefreshControl refreshing={isFetching} onRefresh={refetch} tintColor={colors.accent} />
          }
        >
          <EmptyState
            icon="alert-circle-outline"
            title="Couldn't load this trade"
            hint={(error as Error)?.message || 'Pull down to retry.'}
          />
        </ScrollView>
      ) : (
        <ScrollView
          contentContainerStyle={styles.content}
          refreshControl={
            <RefreshControl refreshing={isFetching} onRefresh={refetch} tintColor={colors.accent} />
          }
        >
          {/* Outcome card */}
          <View style={[styles.outcomeCard, { borderColor: tone.fg + '55', backgroundColor: tone.bg }]}>
            <View style={{ flex: 1 }}>
              <Text style={[styles.outcomeLabel, { color: tone.fg }]}>{tone.label}</Text>
              <Text style={styles.outcomeStrategy} numberOfLines={1}>{data.strategy_name}</Text>
              <Text style={styles.outcomeSub}>
                {data.is_paper ? 'Paper trade' : 'Live trade'} · {data.leverage > 0 ? `${data.leverage}×` : '—'} · fired {shortDate(data.fired_at)}
              </Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
              <Text style={[styles.outcomePnl, { color: tone.fg }]}>{fmtPnl(data.pnl_pct)}</Text>
              <Text style={[styles.outcomePnlUsd, { color: tone.fg }]}>{fmtUsd(data.pnl_usd)}</Text>
            </View>
          </View>

          {/* Why-fired chart */}
          <Text style={styles.sectionLabel}>WHY DID IT FIRE</Text>
          {candles.length > 0 ? (
            <CandleChart
              candles={candles}
              markers={markers}
              priceLines={priceLines}
              width={chartW}
              height={280}
              symbol={data.symbol}
              tf={data.candles?.tf}
              showVolume
              interactive
            />
          ) : (
            <View style={styles.chartEmpty}>
              <Ionicons name="bar-chart-outline" size={22} color={colors.textMute} />
              <Text style={styles.chartEmptyText}>
                Price history isn't available for this symbol on the chart provider.
              </Text>
            </View>
          )}

          {/* Conditions that fired */}
          <Text style={styles.sectionLabel}>CONDITIONS THAT MATCHED</Text>
          <View style={styles.condCard}>
            {data.conditions.length === 0 ? (
              <Text style={styles.condEmpty}>
                No condition snapshot was stored for this trade.
              </Text>
            ) : (
              <View style={styles.condChipRow}>
                {data.conditions.map((c, i) => (
                  <View key={`c-${i}`} style={styles.condChip}>
                    <Ionicons name="checkmark-circle" size={12} color={colors.positive} />
                    <Text style={styles.condChipText} numberOfLines={2}>{c}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>

          {/* Trade plan */}
          <Text style={styles.sectionLabel}>TRADE PLAN</Text>
          <View style={styles.planCard}>
            <PlanRow label="Entry"  value={fmtPx(data.entry_price)} tone="text" />
            <PlanRow label="Take profit" value={fmtPx(data.tp_price)} tone="positive" />
            {data.tp2_price != null ? (
              <PlanRow label="TP2" value={fmtPx(data.tp2_price)} tone="positive" />
            ) : null}
            <PlanRow label="Stop loss" value={fmtPx(data.sl_price)} tone="negative" />
            <PlanRow
              label={data.outcome === 'OPEN' ? 'Current' : 'Exit'}
              value={fmtPx(data.exit_price)}
              tone={data.outcome === 'WIN' ? 'positive' : data.outcome === 'LOSS' ? 'negative' : 'text'}
            />
            <PlanRow label="Position size" value={data.position_size != null ? `$${data.position_size.toFixed(2)}` : '—'} tone="text" />
            <PlanRow label="Closed" value={shortDate(data.closed_at)} tone="text" last />
          </View>

          {data.notes ? (
            <>
              <Text style={styles.sectionLabel}>NOTES</Text>
              <View style={styles.planCard}>
                <Text style={styles.notes}>{data.notes}</Text>
              </View>
            </>
          ) : null}

          <Pressable
            onPress={() => router.push(`/strategy/${data.strategy_id}` as any)}
            style={({ pressed }) => [styles.openStrat, pressed && { opacity: 0.85 }]}
          >
            <Ionicons name="pulse" size={15} color={colors.accent} />
            <Text style={styles.openStratText}>Open parent strategy</Text>
            <Ionicons name="chevron-forward" size={15} color={colors.accent} />
          </Pressable>
        </ScrollView>
      )}
    </View>
  );
}

function PlanRow({
  label, value, tone = 'text', last = false,
}: {
  label: string; value: string; tone?: 'positive' | 'negative' | 'text'; last?: boolean;
}) {
  const color =
    tone === 'positive' ? colors.positive :
    tone === 'negative' ? colors.negative :
    colors.text;
  return (
    <View style={[styles.planRow, !last && styles.planRowDiv]}>
      <Text style={styles.planLabel}>{label}</Text>
      <Text style={[styles.planValue, { color }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  topbar: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  backBtn: {
    width: 32, height: 32, alignItems: 'center', justifyContent: 'center',
    borderRadius: radius.md, backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.border,
  },
  topbarTitle: {
    flex: 1, color: colors.text, fontFamily: font.black, fontSize: 16, letterSpacing: -0.3,
  },

  content: { padding: spacing.lg, paddingBottom: spacing.xxl + 32, gap: spacing.md },

  outcomeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.lg,
    borderRadius: radius.xl,
    borderWidth: 1,
  },
  outcomeLabel: { fontFamily: font.black, fontSize: 11, letterSpacing: 1.2 },
  outcomeStrategy: { color: colors.text, fontFamily: font.black, fontSize: 16, marginTop: 2 },
  outcomeSub: { color: colors.textDim, fontFamily: font.regular, fontSize: 11.5, marginTop: 4 },
  outcomePnl: { fontFamily: font.black, fontSize: 22, fontVariant: ['tabular-nums'] },
  outcomePnlUsd: { fontFamily: font.bold, fontSize: 12, fontVariant: ['tabular-nums'], marginTop: 2 },

  sectionLabel: {
    color: colors.textMute,
    fontFamily: font.black,
    fontSize: 10.5,
    letterSpacing: 1.0,
    marginTop: spacing.sm,
    paddingHorizontal: 2,
  },

  chartEmpty: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    alignItems: 'center',
    gap: 6,
  },
  chartEmptyText: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12.5,
    textAlign: 'center',
  },

  condCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  condChipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  condChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    paddingHorizontal: 9,
    paddingVertical: 5,
    borderRadius: radius.pill,
    backgroundColor: colors.positiveDim,
    borderWidth: 1,
    borderColor: 'rgba(52,211,153,0.32)',
    maxWidth: '100%',
  },
  condChipText: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 11.5,
  },
  condEmpty: { color: colors.textMute, fontFamily: font.regular, fontSize: 12.5 },

  planCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
  },
  planRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 11,
  },
  planRowDiv: { borderBottomWidth: 1, borderBottomColor: colors.divider },
  planLabel: { color: colors.textDim, fontFamily: font.semibold, fontSize: 12.5 },
  planValue: { fontFamily: font.bold, fontSize: 13, fontVariant: ['tabular-nums'] },

  notes: { color: colors.text, fontFamily: font.regular, fontSize: 13.5, lineHeight: 19, padding: spacing.md },

  openStrat: {
    marginTop: spacing.sm,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingVertical: 12,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    backgroundColor: colors.accentDim,
  },
  openStratText: { color: colors.accent, fontFamily: font.black, fontSize: 12.5, letterSpacing: 0.5 },
});
