import React, { useCallback } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, useWindowDimensions } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';

import { Screen } from '@/components/Screen';
import { StatCard } from '@/components/StatCard';
import { EquityCurve } from '@/components/EquityCurve';
import { EmptyState } from '@/components/EmptyState';
import { colors, radius, spacing } from '@/constants/colors';
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

  return (
    <Screen
      title={`Hi, ${user?.first_name || user?.username || 'trader'}`}
      subtitle="Here's how your strategies are doing."
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
          {/* Hero P&L */}
          <View style={styles.hero}>
            <Text style={styles.heroLabel}>TOTAL P&amp;L</Text>
            <Text style={[
              styles.heroValue,
              { color: data.pnl_all > 0 ? colors.positive : data.pnl_all < 0 ? colors.negative : colors.text },
            ]}>
              {fmtPnl(data.pnl_all)}
            </Text>
            <View style={styles.heroFooter}>
              <Ionicons
                name={data.pnl_30d >= 0 ? 'trending-up' : 'trending-down'}
                size={14}
                color={data.pnl_30d >= 0 ? colors.positive : colors.negative}
              />
              <Text style={styles.heroFooterText}>
                {fmtPnl(data.pnl_30d)} in the last 30 days
              </Text>
            </View>
          </View>

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
                hint="Build your first strategy on tradehub.markets to start seeing data here."
              />
            </View>
          )}
        </>
      ) : null}
    </Screen>
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
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.xl,
  },
  heroLabel: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.8,
  },
  heroValue: {
    fontSize: 42,
    fontWeight: '800',
    letterSpacing: -1,
    marginTop: 6,
  },
  heroFooter: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.sm,
  },
  heroFooterText: {
    color: colors.textDim,
    fontSize: 13,
  },
  statRow: {
    flexDirection: 'row',
  },
  section: {
    color: colors.textDim,
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
});
