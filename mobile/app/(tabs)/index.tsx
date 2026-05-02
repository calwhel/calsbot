import React, { useCallback } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, useWindowDimensions } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';

import { Screen } from '@/components/Screen';
import { StatCard } from '@/components/StatCard';
import { EquityCurve } from '@/components/EquityCurve';
import { EmptyState } from '@/components/EmptyState';
import { Logo } from '@/components/Logo';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type Portfolio } from '@/lib/api';

function fmtPnl(v: number): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}
function pnlTone(v: number): 'positive' | 'negative' | 'neutral' {
  if (v > 0.01) return 'positive';
  if (v < -0.01) return 'negative';
  return 'neutral';
}

export default function HomeScreen() {
  const { uid, user } = useAuth();
  const { width } = useWindowDimensions();
  const chartWidth = width - spacing.lg * 2 - spacing.sm * 2;

  const { data, isLoading, isFetching, refetch, isError, error } = useQuery({
    queryKey: ['portfolio', uid],
    queryFn: () => apiGet<Portfolio>('/api/portfolio', uid),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);
  const greetingName = user?.first_name || user?.username || 'trader';

  return (
    <Screen
      title={`Hi, ${greetingName}`}
      subtitle="Here's how your strategies are doing today."
      rightSlot={<Logo size={42} />}
      refreshing={isFetching && !isLoading}
      onRefresh={onRefresh}
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
        <>
          {/* Hero P&L card with gradient + glow */}
          <HeroCard pnlAll={data.pnl_all} pnl30d={data.pnl_30d} />

          {/* Equity curve */}
          <View style={{ marginTop: spacing.lg }}>
            <SectionHeader label="30-day equity curve" />
            <EquityCurve values={data.equity_30d?.values || []} width={chartWidth} height={160} />
          </View>

          {/* Stat grid */}
          <View style={[styles.statRow, { marginTop: spacing.lg }]}>
            <StatCard label="7-day P&L" value={fmtPnl(data.pnl_7d)} tone={pnlTone(data.pnl_7d)} />
            <View style={{ width: spacing.md }} />
            <StatCard label="Win rate" value={`${data.win_rate.toFixed(1)}%`} sub={`${data.total_trades} closed trades`} />
          </View>
          <View style={[styles.statRow, { marginTop: spacing.md }]}>
            <StatCard label="Active strategies" value={`${data.active_count}`} sub={`of ${data.total_strategies} total`} tone="accent" />
            <View style={{ width: spacing.md }} />
            <StatCard label="Open positions" value={`${data.open_trades}`} sub={data.open_trades > 0 ? 'Live now' : 'None right now'} />
          </View>

          {data.total_strategies === 0 && (
            <View style={{ marginTop: spacing.xl }}>
              <EmptyState
                icon="rocket-outline"
                title="No strategies yet"
                hint="Build your first strategy to start seeing data here."
                tone="accent"
              />
            </View>
          )}
        </>
      ) : null}
    </Screen>
  );
}

function HeroCard({ pnlAll, pnl30d }: { pnlAll: number; pnl30d: number }) {
  const isPos = pnlAll >= 0;
  const valueColor = pnlAll > 0 ? colors.positive : pnlAll < 0 ? colors.negative : colors.text;
  const uid = React.useId().replace(/:/g, '');
  const bgId = `hero-bg-${uid}`;
  const shineId = `hero-shine-${uid}`;

  return (
    <View style={[styles.hero, glow.accent]}>
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <SvgLinearGradient id={bgId} x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#1a2452" />
            <Stop offset="0.55" stopColor="#13193a" />
            <Stop offset="1" stopColor="#0a1024" />
          </SvgLinearGradient>
          <SvgLinearGradient id={shineId} x1="0" y1="0" x2="1" y2="0">
            <Stop offset="0" stopColor="#22d3ee" stopOpacity="0.7" />
            <Stop offset="0.6" stopColor="#3b82f6" stopOpacity="0.3" />
            <Stop offset="1" stopColor="#22d3ee" stopOpacity="0" />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill={`url(#${bgId})`} />
        <Rect width="100%" height="3" fill={`url(#${shineId})`} />
      </Svg>
      <View style={styles.heroInner}>
        <View style={styles.heroLabelRow}>
          <Text style={styles.heroLabel}>TOTAL P&amp;L</Text>
          <View style={styles.heroBadge}>
            <Ionicons
              name={pnl30d >= 0 ? 'trending-up' : 'trending-down'}
              size={12}
              color={pnl30d >= 0 ? colors.positive : colors.negative}
            />
            <Text style={[
              styles.heroBadgeText,
              { color: pnl30d >= 0 ? colors.positive : colors.negative },
            ]}>
              {fmtPnl(pnl30d)} · 30d
            </Text>
          </View>
        </View>
        <Text style={[styles.heroValue, { color: valueColor }]}>{fmtPnl(pnlAll)}</Text>
        <Text style={styles.heroFootnote}>
          {isPos ? 'You are in the green across all-time trades.' : 'Long-term P&L is below zero — review under-performers.'}
        </Text>
      </View>
    </View>
  );
}

function SectionHeader({ label }: { label: string }) {
  return <Text style={styles.section}>{label}</Text>;
}

const styles = StyleSheet.create({
  loading: {
    paddingVertical: spacing.xxl + spacing.lg,
    alignItems: 'center',
  },
  hero: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.18)',
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  heroInner: {
    padding: spacing.xl,
  },
  heroLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  heroLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 1.0,
  },
  heroBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: 'rgba(0,0,0,0.32)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.06)',
  },
  heroBadgeText: {
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.3,
  },
  heroValue: {
    fontFamily: font.black,
    fontSize: 52,
    letterSpacing: -1.6,
    marginTop: 10,
    fontVariant: ['tabular-nums'],
  },
  heroFootnote: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    marginTop: spacing.sm,
    lineHeight: 18,
  },
  statRow: {
    flexDirection: 'row',
  },
  section: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
});
