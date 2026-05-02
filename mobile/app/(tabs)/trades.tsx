import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SectionList,
  Pressable,
  ActivityIndicator,
  RefreshControl,
  ScrollView,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { AmbientBg } from '@/components/AmbientBg';
import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { CoinChip } from '@/components/CoinChip';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import {
  apiGet,
  type PortfolioTradesResponse,
  type PortfolioTrade,
  type TradeFilter,
} from '@/lib/api';

// ─── Helpers ───────────────────────────────────────────────────────────────

function fmtPnl(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function fmtPrice(v: number | null | undefined): string {
  if (v === null || v === undefined) return '—';
  if (v >= 1000) return v.toFixed(0);
  if (v >= 1)    return v.toFixed(2);
  if (v >= 0.01) return v.toFixed(4);
  return v.toFixed(6);
}

function fmtDuration(mins: number | null | undefined): string {
  if (!mins) return '—';
  if (mins < 60)        return `${mins}m`;
  if (mins < 60 * 24)   return `${(mins / 60).toFixed(1)}h`;
  return `${(mins / (60 * 24)).toFixed(1)}d`;
}

function dayKey(iso: string | null): string {
  if (!iso) return 'Pending';
  try {
    const d = new Date(iso);
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const yest = new Date(today.getTime() - 86400000);
    const dDay = new Date(d);
    dDay.setHours(0, 0, 0, 0);
    if (dDay.getTime() === today.getTime()) return 'Today';
    if (dDay.getTime() === yest.getTime())  return 'Yesterday';
    const opts: Intl.DateTimeFormatOptions = { month: 'short', day: 'numeric', weekday: 'short' };
    if (d.getFullYear() !== today.getFullYear()) opts.year = 'numeric';
    return d.toLocaleDateString(undefined, opts);
  } catch {
    return 'Unknown';
  }
}

function timeOfDay(iso: string | null): string {
  if (!iso) return '';
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  } catch { return ''; }
}

// ─── Filter chips ──────────────────────────────────────────────────────────

const FILTERS: Array<{ key: TradeFilter; label: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { key: 'all',    label: 'All',     icon: 'apps-outline' },
  { key: 'open',   label: 'Open',    icon: 'flash-outline' },
  { key: 'wins',   label: 'Wins',    icon: 'trending-up' },
  { key: 'losses', label: 'Losses',  icon: 'trending-down' },
  { key: 'live',   label: 'Live',    icon: 'pulse' },
  { key: 'paper',  label: 'Paper',   icon: 'document-text-outline' },
];

function FilterChips({
  value,
  onChange,
}: {
  value: TradeFilter;
  onChange: (f: TradeFilter) => void;
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
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

// ─── Trade row ─────────────────────────────────────────────────────────────

const TradeRow = React.memo(function TradeRow({ t, onPress }: { t: PortfolioTrade; onPress: () => void }) {
  const pnl =
    t.outcome === 'OPEN' && t.unrealised_pnl !== undefined && t.unrealised_pnl !== null
      ? t.unrealised_pnl
      : t.pnl_pct;
  const pnlColor =
    pnl === null || pnl === undefined
      ? colors.textDim
      : pnl > 0
        ? colors.positive
        : pnl < 0
          ? colors.negative
          : colors.text;

  const isOpen = t.outcome === 'OPEN';
  const isLong = t.direction === 'LONG';

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        pressed && { opacity: 0.85, transform: [{ scale: 0.997 }] },
      ]}
    >
      {/* Direction stripe */}
      <View
        style={[
          styles.dirStripe,
          { backgroundColor: isLong ? colors.positive : colors.negative },
        ]}
      />

      {/* Coin badge */}
      <View style={styles.coinWrap}>
        <CoinChip symbol={t.symbol} size={36} />
      </View>

      <View style={{ flex: 1 }}>
        {/* Top row: symbol + outcome pill + pnl */}
        <View style={styles.rowTop}>
          <View style={styles.symbolWrap}>
            <Text style={styles.symbol} numberOfLines={1}>{t.symbol.replace('USDT', '')}</Text>
            <Text style={styles.symbolBase}>USDT</Text>
            <View
              style={[
                styles.dirBadge,
                { backgroundColor: isLong ? colors.positiveDim : colors.negativeDim },
              ]}
            >
              <Ionicons
                name={isLong ? 'arrow-up' : 'arrow-down'}
                size={10}
                color={isLong ? colors.positive : colors.negative}
              />
              <Text
                style={[
                  styles.dirBadgeText,
                  { color: isLong ? colors.positive : colors.negative },
                ]}
              >
                {t.direction}
              </Text>
            </View>
          </View>
          <Text style={[styles.pnl, { color: pnlColor }]}>{fmtPnl(pnl)}</Text>
        </View>

        {/* Strategy name */}
        <View style={styles.metaRow}>
          <Ionicons name="pulse" size={11} color={colors.textMute} />
          <Text style={styles.strategyName} numberOfLines={1}>{t.strategy_name}</Text>
          <View style={styles.dotSep} />
          <Pill
            label={isOpen ? 'OPEN' : t.outcome}
            tone={
              isOpen
                ? 'accent'
                : t.outcome === 'WIN'
                  ? 'positive'
                  : 'negative'
            }
            small
          />
          {t.is_paper ? (
            <>
              <View style={styles.dotSep} />
              <Pill label="paper" tone="neutral" small />
            </>
          ) : null}
        </View>

        {/* Bottom row: prices + duration + time */}
        <View style={styles.bottomRow}>
          <View style={styles.priceCell}>
            <Text style={styles.priceLabel}>ENTRY</Text>
            <Text style={styles.priceVal}>{fmtPrice(t.entry_price)}</Text>
          </View>
          <Ionicons
            name="arrow-forward"
            size={11}
            color={colors.textMute}
            style={{ marginHorizontal: 4 }}
          />
          <View style={styles.priceCell}>
            <Text style={styles.priceLabel}>{isOpen ? 'NOW' : 'EXIT'}</Text>
            <Text style={styles.priceVal}>
              {isOpen ? fmtPrice(t.live_price) : fmtPrice(t.exit_price)}
            </Text>
          </View>
          {t.leverage ? (
            <View style={styles.lev}>
              <Text style={styles.levText}>{t.leverage}x</Text>
            </View>
          ) : null}
          <View style={{ flex: 1 }} />
          {!isOpen && t.duration_mins ? (
            <View style={styles.timeBlock}>
              <Ionicons name="time-outline" size={11} color={colors.textMute} />
              <Text style={styles.timeText}>{fmtDuration(t.duration_mins)}</Text>
            </View>
          ) : null}
          <Text style={styles.timeStamp}>{timeOfDay(t.closed_at || t.fired_at)}</Text>
        </View>
      </View>
    </Pressable>
  );
});

// ─── Section header ────────────────────────────────────────────────────────

function SectionHeader({
  title,
  count,
  pnl,
}: {
  title: string;
  count: number;
  pnl: number;
}) {
  const tone = pnl > 0.01 ? colors.positive : pnl < -0.01 ? colors.negative : colors.textDim;
  return (
    <View style={styles.sectionHeader}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <View style={styles.sectionMeta}>
        <Text style={styles.sectionCount}>{count} trade{count === 1 ? '' : 's'}</Text>
        <View style={styles.sectionDot} />
        <Text style={[styles.sectionPnl, { color: tone }]}>{fmtPnl(pnl)}</Text>
      </View>
    </View>
  );
}

// ─── Screen ────────────────────────────────────────────────────────────────

export default function TradesScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const [filter, setFilter] = useState<TradeFilter>('all');

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['portfolio-trades', uid, filter],
    queryFn: () =>
      apiGet<PortfolioTradesResponse>('/api/portfolio/trades', uid, {
        limit: 200,
        filter,
      }),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  // Group by date for the SectionList
  const sections = useMemo(() => {
    const trades = data?.trades || [];
    const groups: Record<string, PortfolioTrade[]> = {};
    const order: string[] = [];
    for (const t of trades) {
      const key = dayKey(t.closed_at || t.fired_at);
      if (!(key in groups)) {
        groups[key] = [];
        order.push(key);
      }
      groups[key].push(t);
    }
    return order.map((title) => {
      const items = groups[title];
      const pnl = items.reduce((sum, t) => {
        const v =
          t.outcome === 'OPEN' && t.unrealised_pnl !== undefined && t.unrealised_pnl !== null
            ? t.unrealised_pnl
            : (t.pnl_pct ?? 0);
        return sum + (v ?? 0);
      }, 0);
      return { title, count: items.length, pnl, data: items };
    });
  }, [data]);

  // Top header summary (visible across all filters)
  const topSummary = useMemo(() => {
    const trades = data?.trades || [];
    let wins = 0, losses = 0, openCount = 0, totalPnl = 0;
    for (const t of trades) {
      if (t.outcome === 'WIN')  wins++;
      if (t.outcome === 'LOSS') losses++;
      if (t.outcome === 'OPEN') openCount++;
      const v =
        t.outcome === 'OPEN' && t.unrealised_pnl !== undefined && t.unrealised_pnl !== null
          ? t.unrealised_pnl
          : (t.pnl_pct ?? 0);
      totalPnl += v ?? 0;
    }
    const closed = wins + losses;
    const wr = closed > 0 ? (wins / closed) * 100 : 0;
    return { wins, losses, openCount, totalPnl, wr, total: trades.length };
  }, [data]);

  const Header = (
    <View style={styles.header}>
      <View style={styles.headerRow}>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Trades</Text>
          <Text style={styles.subtitle}>
            {data?.total
              ? `${data.total} total · across all your strategies`
              : 'Every fill from every strategy you own'}
          </Text>
        </View>
        <View
          style={[
            styles.headerStat,
            {
              borderColor:
                topSummary.totalPnl >= 0
                  ? 'rgba(52,211,153,0.32)'
                  : 'rgba(248,113,113,0.32)',
            },
          ]}
        >
          <Text style={styles.headerStatLabel}>P&L (page)</Text>
          <Text
            style={[
              styles.headerStatValue,
              {
                color:
                  topSummary.totalPnl > 0.01
                    ? colors.positive
                    : topSummary.totalPnl < -0.01
                      ? colors.negative
                      : colors.text,
              },
            ]}
          >
            {fmtPnl(topSummary.totalPnl)}
          </Text>
        </View>
      </View>

      {/* Mini stat rail */}
      <View style={styles.miniStats}>
        <View style={styles.miniStat}>
          <Text style={styles.miniStatValue}>{topSummary.wins}</Text>
          <Text style={[styles.miniStatLabel, { color: colors.positive }]}>WINS</Text>
        </View>
        <View style={styles.miniStatSep} />
        <View style={styles.miniStat}>
          <Text style={styles.miniStatValue}>{topSummary.losses}</Text>
          <Text style={[styles.miniStatLabel, { color: colors.negative }]}>LOSSES</Text>
        </View>
        <View style={styles.miniStatSep} />
        <View style={styles.miniStat}>
          <Text style={styles.miniStatValue}>{topSummary.openCount}</Text>
          <Text style={[styles.miniStatLabel, { color: colors.accent }]}>OPEN</Text>
        </View>
        <View style={styles.miniStatSep} />
        <View style={styles.miniStat}>
          <Text style={styles.miniStatValue}>
            {topSummary.wins + topSummary.losses > 0 ? `${topSummary.wr.toFixed(0)}%` : '—'}
          </Text>
          <Text style={styles.miniStatLabel}>WIN RATE</Text>
        </View>
      </View>

      <FilterChips value={filter} onChange={setFilter} />
    </View>
  );

  if (isLoading) {
    return (
      <View style={[styles.root, { paddingTop: insets.top }]}>
        <AmbientBg variant="duo" />
        {Header}
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} size="large" />
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
          <EmptyState
            icon="cloud-offline-outline"
            title="Couldn't load your trades"
            hint="Pull down to retry."
          />
        </ScrollView>
      </View>
    );
  }

  if (!data || data.trades.length === 0) {
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
          <EmptyState
            icon="time-outline"
            title={filter === 'all' ? 'No trades yet' : 'No trades match this filter'}
            hint={
              filter === 'all'
                ? 'Activate a strategy from the Strategies tab and your fills will land here.'
                : 'Try another filter — your other trades are still listed.'
            }
            tone="accent"
          />
        </ScrollView>
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <AmbientBg variant="duo" />
      <SectionList
        sections={sections}
        keyExtractor={(t, i) => `t-${t.id}-${i}`}
        renderSectionHeader={({ section }) => (
          <SectionHeader
            title={section.title}
            count={(section as any).count}
            pnl={(section as any).pnl}
          />
        )}
        renderItem={({ item }) => (
          <TradeRow
            t={item}
            onPress={() => {
              Haptics.selectionAsync().catch(() => {});
              router.push(`/strategy/${item.strategy_id}`);
            }}
          />
        )}
        ListHeaderComponent={Header}
        ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        SectionSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
        contentContainerStyle={styles.listContent}
        stickySectionHeadersEnabled={false}
        refreshControl={
          <RefreshControl
            refreshing={isFetching && !isLoading}
            onRefresh={onRefresh}
            tintColor={colors.accent}
            colors={[colors.accent]}
            progressBackgroundColor={colors.bgElev}
          />
        }
        showsVerticalScrollIndicator={false}
      />
    </View>
  );
}

// ─── Styles ────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center' },

  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  title: { color: colors.text, fontFamily: font.black, fontSize: 30, letterSpacing: -0.8 },
  subtitle: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13.5,
    marginTop: 4,
    lineHeight: 18,
  },
  headerStat: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderRadius: radius.lg,
    paddingHorizontal: spacing.md,
    paddingVertical: 8,
    alignItems: 'flex-end',
    minWidth: 96,
  },
  headerStatLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
  },
  headerStatValue: {
    fontFamily: font.black,
    fontSize: 17,
    fontVariant: ['tabular-nums'],
    marginTop: 2,
  },

  miniStats: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    marginTop: spacing.lg,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.md,
    ...glow.card,
  },
  miniStat: {
    flex: 1,
    alignItems: 'center',
  },
  miniStatValue: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 18,
    letterSpacing: -0.4,
    fontVariant: ['tabular-nums'],
  },
  miniStatLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.7,
    marginTop: 2,
  },
  miniStatSep: {
    width: 1,
    backgroundColor: colors.border,
    marginVertical: 4,
  },

  chipRow: {
    paddingTop: spacing.lg,
    paddingBottom: 4,
    gap: 8,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
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
  chipTextActive: {
    color: colors.accentText,
  },

  listContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl + 96,
  },

  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: spacing.lg,
    paddingBottom: spacing.sm,
  },
  sectionTitle: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 13,
    letterSpacing: 0.3,
  },
  sectionMeta: { flexDirection: 'row', alignItems: 'center' },
  sectionCount: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
  },
  sectionDot: {
    width: 3,
    height: 3,
    borderRadius: 1.5,
    backgroundColor: colors.textMute,
    marginHorizontal: 6,
  },
  sectionPnl: {
    fontFamily: font.bold,
    fontSize: 11,
    fontVariant: ['tabular-nums'],
    letterSpacing: 0.3,
  },

  row: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    overflow: 'hidden',
  },
  dirStripe: {
    width: 3,
    alignSelf: 'stretch',
    borderRadius: 2,
    marginRight: spacing.sm,
  },
  coinWrap: {
    marginRight: spacing.md,
    alignSelf: 'center',
  },
  rowTop: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  symbolWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    flex: 1,
  },
  symbol: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 16,
    letterSpacing: -0.3,
  },
  symbolBase: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 11,
    marginRight: 6,
  },
  dirBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.pill,
  },
  dirBadgeText: {
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.5,
  },
  pnl: {
    fontFamily: font.black,
    fontSize: 17,
    letterSpacing: -0.3,
    fontVariant: ['tabular-nums'],
  },

  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: 6,
  },
  strategyName: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 12,
    flexShrink: 1,
  },
  dotSep: {
    width: 3,
    height: 3,
    borderRadius: 1.5,
    backgroundColor: colors.textMute,
  },

  bottomRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 8,
  },
  priceCell: {},
  priceLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 8,
    letterSpacing: 0.7,
  },
  priceVal: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 12,
    fontVariant: ['tabular-nums'],
  },
  lev: {
    marginLeft: 8,
    paddingHorizontal: 6,
    paddingVertical: 2,
    backgroundColor: colors.warningDim,
    borderRadius: radius.sm,
  },
  levText: {
    color: colors.warning,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.3,
  },
  timeBlock: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 3,
    marginRight: 8,
  },
  timeText: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 11,
  },
  timeStamp: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
    fontVariant: ['tabular-nums'],
  },
});
