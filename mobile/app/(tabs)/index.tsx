import React, { useCallback, useMemo } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  ScrollView,
  Pressable,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';

import { Screen } from '@/components/Screen';
import { MeshHero } from '@/components/MeshHero';
import { BentoTile } from '@/components/BentoTile';
import { ActionTile } from '@/components/ActionTile';
import { SectionLabel } from '@/components/SectionLabel';
import { MiniDonut } from '@/components/MiniDonut';
import { CoinChip } from '@/components/CoinChip';
import { EmptyState } from '@/components/EmptyState';
import { Logo } from '@/components/Logo';
import { QuickstartCard } from '@/components/QuickstartCard';
import { Pill } from '@/components/Pill';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import {
  apiGet,
  type Portfolio,
  type Strategy,
  type PortfolioTradesResponse,
  type PortfolioTrade,
} from '@/lib/api';

function fmtPnl(v: number): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}
function pnlTone(v: number): 'positive' | 'negative' | 'neutral' {
  if (v > 0.01) return 'positive';
  if (v < -0.01) return 'negative';
  return 'neutral';
}
function timeAgo(iso: string | null): string {
  if (!iso) return '';
  try {
    const ms = Date.now() - new Date(iso).getTime();
    if (ms < 60_000)         return 'just now';
    if (ms < 3_600_000)      return `${Math.floor(ms / 60_000)}m`;
    if (ms < 86_400_000)     return `${Math.floor(ms / 3_600_000)}h`;
    if (ms < 7 * 86_400_000) return `${Math.floor(ms / 86_400_000)}d`;
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch { return ''; }
}

export default function HomeScreen() {
  const { uid, user } = useAuth();
  const router = useRouter();

  const portfolioQ = useQuery({
    queryKey: ['portfolio', uid],
    queryFn: () => apiGet<Portfolio>('/api/portfolio', uid),
    enabled: !!uid,
  });

  const strategiesQ = useQuery({
    queryKey: ['strategies', uid],
    queryFn: () => apiGet<Strategy[]>('/api/strategies', uid),
    enabled: !!uid,
  });

  const recentTradesQ = useQuery({
    queryKey: ['portfolio-trades', uid, 'recent-home'],
    queryFn: () =>
      apiGet<PortfolioTradesResponse>('/api/portfolio/trades', uid, { limit: 4, filter: 'all' }),
    enabled: !!uid,
  });

  const data = portfolioQ.data;
  const isLoading = portfolioQ.isLoading;
  const isFetching =
    portfolioQ.isFetching || strategiesQ.isFetching || recentTradesQ.isFetching;
  const isError = portfolioQ.isError;
  const error = portfolioQ.error;

  const onRefresh = useCallback(() => {
    portfolioQ.refetch();
    strategiesQ.refetch();
    recentTradesQ.refetch();
  }, [portfolioQ, strategiesQ, recentTradesQ]);

  const greetingName = user?.first_name || user?.username || 'trader';
  const greetingTime = useMemo(() => {
    const h = new Date().getHours();
    if (h < 5)  return 'Late night';
    if (h < 12) return 'Good morning';
    if (h < 17) return 'Good afternoon';
    if (h < 21) return 'Good evening';
    return 'Late night';
  }, []);

  const topStrategies = useMemo(() => {
    const list = strategiesQ.data || [];
    return [...list]
      .filter((s) => (s.performance?.total_trades ?? 0) > 0)
      .sort((a, b) => (b.performance?.total_pnl ?? 0) - (a.performance?.total_pnl ?? 0))
      .slice(0, 3);
  }, [strategiesQ.data]);

  return (
    <Screen
      title={`${greetingTime}, ${greetingName}`}
      subtitle="Here's a snapshot of your trading desk."
      rightSlot={<Logo size={42} />}
      refreshing={isFetching && !isLoading}
      onRefresh={onRefresh}
      ambient="duo"
    >
      {isLoading ? (
        <View style={styles.loading}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      ) : isError ? (
        <EmptyState
          icon="cloud-offline-outline"
          title="Couldn't load your portfolio"
          hint={(error as Error)?.message || 'Pull down to retry.'}
        />
      ) : data ? (
        data.total_strategies === 0 ? (
          <QuickstartCard onStart={() => router.push('/build')} />
        ) : (
          <>
            {/* Premium hero with embedded equity sparkline */}
            <MeshHero
              label="TOTAL P&L"
              value={fmtPnl(data.pnl_all)}
              badgeText={`${fmtPnl(data.pnl_30d)} · 30d`}
              badgeTone={data.pnl_30d > 0.01 ? 'positive' : data.pnl_30d < -0.01 ? 'negative' : 'neutral'}
              footnote={
                data.pnl_all > 0
                  ? 'You are in the green across all strategies.'
                  : data.pnl_all < 0
                  ? 'Long-term P&L is negative — review under-performers below.'
                  : "Let's get the first trades on the board."
              }
              spark={data.equity_30d?.values || []}
              tone={pnlTone(data.pnl_all)}
            />

            {/* Quick actions */}
            <View style={styles.quickWrap}>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.quickRow}
              >
                <ActionTile
                  icon="add"
                  label="New"
                  tone="accent"
                  onPress={() => router.push('/build' as any)}
                />
                <ActionTile
                  icon="pulse"
                  label="Strategies"
                  tone="violet"
                  onPress={() => router.push('/(tabs)/strategies' as any)}
                />
                <ActionTile
                  icon="swap-horizontal"
                  label="Trades"
                  tone="positive"
                  onPress={() => router.push('/(tabs)/trades' as any)}
                />
                <ActionTile
                  icon="storefront"
                  label="Market"
                  tone="gold"
                  onPress={() => router.push('/(tabs)/marketplace' as any)}
                />
                <ActionTile
                  icon="person-circle"
                  label="Account"
                  tone="magenta"
                  onPress={() => router.push('/(tabs)/settings' as any)}
                />
              </ScrollView>
            </View>

            {/* Bento grid — 2 rows × 2 tiles */}
            <View style={styles.bentoSection}>
              <SectionLabel label="At a glance" caption="Key numbers across all strategies" />
              <View style={styles.bentoRow}>
                <BentoTile
                  icon="trending-up"
                  label="7-day P&L"
                  value={fmtPnl(data.pnl_7d)}
                  tone={data.pnl_7d > 0 ? 'positive' : data.pnl_7d < 0 ? 'negative' : 'neutral'}
                  sub={data.pnl_7d > 0 ? 'Up week-on-week' : data.pnl_7d < 0 ? 'Down this week' : 'Flat this week'}
                />
                <View style={{ width: spacing.md }} />
                <BentoTile
                  icon="trophy-outline"
                  label="Win rate"
                  value={`${data.win_rate.toFixed(1)}%`}
                  tone={data.win_rate >= 55 ? 'positive' : data.win_rate < 40 ? 'negative' : 'warning'}
                  sub={`${data.total_trades} closed trades`}
                />
              </View>
              <View style={[styles.bentoRow, { marginTop: spacing.md }]}>
                <BentoTile
                  icon="flash"
                  label="Active"
                  value={`${data.active_count}`}
                  tone="accent"
                  sub={`of ${data.total_strategies} strateg${data.total_strategies === 1 ? 'y' : 'ies'}`}
                  onPress={() => router.push('/(tabs)/strategies' as any)}
                />
                <View style={{ width: spacing.md }} />
                <BentoTile
                  icon="radio-outline"
                  label="Open positions"
                  value={`${data.open_trades}`}
                  tone={data.open_trades > 0 ? 'mint' : 'neutral'}
                  sub={data.open_trades > 0 ? 'Live in the market' : 'Nothing open right now'}
                  onPress={() => router.push('/(tabs)/trades' as any)}
                />
              </View>
            </View>

            {/* Top strategies leaderboard */}
            {topStrategies.length > 0 ? (
              <View style={styles.leaderSection}>
                <SectionLabel
                  label="Top performers"
                  tone="positive"
                  actionLabel="All"
                  onActionPress={() => router.push('/(tabs)/strategies' as any)}
                />
                <View style={styles.leaderCard}>
                  {topStrategies.map((s, idx) => {
                    const pnl = s.performance.total_pnl ?? 0;
                    const wr = s.performance.win_rate ?? 0;
                    const trades = s.performance.total_trades ?? 0;
                    const symbol = (s.config?.symbol as string) || 'BTC';
                    return (
                      <Pressable
                        key={`top-${s.id}`}
                        onPress={() => router.push(`/strategy/${s.id}`)}
                        style={({ pressed }) => [
                          styles.leaderRow,
                          idx > 0 && styles.leaderRowDiv,
                          pressed && { opacity: 0.85 },
                        ]}
                      >
                        <View style={styles.rank}>
                          <Text style={styles.rankText}>{idx + 1}</Text>
                        </View>
                        <CoinChip symbol={symbol} size={32} />
                        <View style={{ flex: 1, marginLeft: spacing.md }}>
                          <Text style={styles.leaderName} numberOfLines={1}>{s.name}</Text>
                          <Text style={styles.leaderSub} numberOfLines={1}>
                            {symbol} · {trades} trade{trades === 1 ? '' : 's'}
                          </Text>
                        </View>
                        <View style={{ alignItems: 'flex-end', marginRight: spacing.sm }}>
                          <Text
                            style={[
                              styles.leaderPnl,
                              { color: pnl > 0 ? colors.positive : pnl < 0 ? colors.negative : colors.text },
                            ]}
                          >
                            {fmtPnl(pnl)}
                          </Text>
                          <Text style={styles.leaderWr}>{wr.toFixed(0)}% win</Text>
                        </View>
                        <MiniDonut value={wr} size={36} />
                      </Pressable>
                    );
                  })}
                </View>
              </View>
            ) : null}

            {/* Recent activity */}
            {(recentTradesQ.data?.trades?.length ?? 0) > 0 ? (
              <View style={styles.activitySection}>
                <SectionLabel
                  label="Recent activity"
                  tone="violet"
                  actionLabel="See all"
                  onActionPress={() => router.push('/(tabs)/trades' as any)}
                />
                <View style={styles.activityCard}>
                  {(recentTradesQ.data!.trades).slice(0, 4).map((t, idx) => (
                    <RecentTradeRow
                      key={`rec-${t.id}-${idx}`}
                      t={t}
                      isFirst={idx === 0}
                      onPress={() => router.push(`/strategy/${t.strategy_id}`)}
                    />
                  ))}
                </View>
              </View>
            ) : null}

            {/* Inactive strategies tip */}
            {data.active_count === 0 ? (
              <View style={[styles.tipBanner, { marginTop: spacing.lg }]}>
                <Ionicons name="information-circle" size={18} color={colors.warning} />
                <Text style={styles.tipText}>
                  None of your strategies are active right now. Open one and tap “Activate” to start collecting trades.
                </Text>
              </View>
            ) : null}
          </>
        )
      ) : null}
    </Screen>
  );
}

function RecentTradeRow({
  t,
  isFirst,
  onPress,
}: {
  t: PortfolioTrade;
  isFirst: boolean;
  onPress: () => void;
}) {
  const isLong = t.direction === 'LONG';
  const isOpen = t.outcome === 'OPEN';
  const pnl = isOpen ? (t.unrealised_pnl ?? null) : t.pnl_pct;
  const pnlColor = pnl == null
    ? colors.textDim
    : pnl > 0 ? colors.positive : pnl < 0 ? colors.negative : colors.text;

  const uid = React.useId().replace(/:/g, '');
  const stripeId = `act-stripe-${uid}`;

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.activityRow,
        !isFirst && styles.activityRowDiv,
        pressed && { opacity: 0.85 },
      ]}
    >
      <CoinChip symbol={t.symbol} size={34} />
      <View style={{ flex: 1, marginLeft: spacing.md }}>
        <View style={styles.activityRowTop}>
          <Text style={styles.activitySymbol} numberOfLines={1}>
            {t.symbol.replace('USDT', '')}
            <Text style={styles.activitySymbolBase}>USDT</Text>
          </Text>
          <View
            style={[
              styles.dirChip,
              {
                backgroundColor: isLong ? colors.positiveDim : colors.negativeDim,
                borderColor: isLong ? 'rgba(52,211,153,0.32)' : 'rgba(248,113,113,0.32)',
              },
            ]}
          >
            <Ionicons
              name={isLong ? 'arrow-up' : 'arrow-down'}
              size={9}
              color={isLong ? colors.positive : colors.negative}
            />
            <Text
              style={[
                styles.dirChipText,
                { color: isLong ? colors.positive : colors.negative },
              ]}
            >
              {t.direction}
            </Text>
          </View>
          {isOpen ? (
            <View style={styles.openDot}>
              <Svg width={6} height={6}>
                <Defs>
                  <SvgLinearGradient id={stripeId} x1="0" y1="0" x2="1" y2="1">
                    <Stop offset="0" stopColor={colors.accent} />
                    <Stop offset="1" stopColor={colors.accentSoft} />
                  </SvgLinearGradient>
                </Defs>
                <Rect width="6" height="6" rx="3" ry="3" fill={`url(#${stripeId})`} />
              </Svg>
            </View>
          ) : null}
        </View>
        <Text style={styles.activityStrategy} numberOfLines={1}>
          {t.strategy_name} · {timeAgo(t.closed_at || t.fired_at)}
        </Text>
      </View>
      <View style={{ alignItems: 'flex-end' }}>
        <Text style={[styles.activityPnl, { color: pnlColor }]}>
          {pnl != null ? fmtPnl(pnl) : '—'}
        </Text>
        {isOpen ? (
          <Text style={styles.activityOpen}>OPEN</Text>
        ) : (
          <Text style={[styles.activityOutcome, { color: t.outcome === 'WIN' ? colors.positive : colors.negative }]}>
            {t.outcome}
          </Text>
        )}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  loading: {
    paddingVertical: spacing.xxl + spacing.lg,
    alignItems: 'center',
  },
  quickWrap: {
    marginTop: spacing.lg,
    marginHorizontal: -spacing.lg,
  },
  quickRow: {
    paddingHorizontal: spacing.lg,
    gap: spacing.xs,
  },
  bentoSection: { marginTop: spacing.xl },
  bentoRow: { flexDirection: 'row' },
  leaderSection: { marginTop: spacing.xl },
  leaderCard: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    ...glow.card,
  },
  leaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
    gap: 6,
  },
  leaderRowDiv: {
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  rank: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: colors.bgElev,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 4,
  },
  rankText: {
    color: colors.textDim,
    fontFamily: font.black,
    fontSize: 11,
    letterSpacing: -0.4,
  },
  leaderName: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14.5,
    letterSpacing: -0.2,
  },
  leaderSub: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11.5,
    marginTop: 2,
  },
  leaderPnl: {
    fontFamily: font.black,
    fontSize: 15,
    fontVariant: ['tabular-nums'],
    letterSpacing: -0.3,
  },
  leaderWr: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 10.5,
    marginTop: 1,
  },
  activitySection: { marginTop: spacing.xl },
  activityCard: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
    ...glow.card,
  },
  activityRow: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: spacing.md,
  },
  activityRowDiv: {
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  activityRowTop: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  activitySymbol: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 14.5,
    letterSpacing: -0.3,
  },
  activitySymbolBase: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10.5,
  },
  dirChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  dirChipText: {
    fontFamily: font.black,
    fontSize: 9,
    letterSpacing: 0.5,
  },
  openDot: {
    marginLeft: 2,
  },
  activityStrategy: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11.5,
    marginTop: 3,
  },
  activityPnl: {
    fontFamily: font.black,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
    letterSpacing: -0.2,
  },
  activityOutcome: {
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.6,
    marginTop: 1,
  },
  activityOpen: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.6,
    marginTop: 1,
  },
  tipBanner: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: 'rgba(245,158,11,0.08)',
    borderColor: 'rgba(245,158,11,0.28)',
    borderWidth: 1,
    borderRadius: radius.lg,
    padding: spacing.md,
  },
  tipText: {
    flex: 1,
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 12.5,
    lineHeight: 17,
  },
});
