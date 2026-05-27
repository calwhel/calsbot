import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  Pressable,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';

import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet } from '@/lib/api';

// ─── Types ──────────────────────────────────────────────────────────────────

type ScannerSignal = {
  pair:      string;
  timeframe: string;
  signal:    'BOS' | 'CHoCH' | 'FVG' | 'OB' | string;
  direction: 'LONG' | 'SHORT';
  desc:      string;
  price:     number;
  detail:    string;
  sig_key:   string;
};

type ScannerResponse = {
  signals:       ScannerSignal[];
  count:         number;
  pairs_scanned: number;
  scanned_at:    string;
  cache_ttl:     number;
};

// ─── Filter types ──────────────────────────────────────────────────────────

type SignalFilter = 'ALL' | 'CHoCH' | 'BOS' | 'FVG' | 'OB';
type TfFilter    = 'ALL' | '15m' | '1h';
type DirFilter   = 'ALL' | 'LONG' | 'SHORT';

// ─── Signal badge colours ──────────────────────────────────────────────────

const SIGNAL_COLORS: Record<string, { bg: string; text: string }> = {
  CHoCH: { bg: 'rgba(214,163,92,0.15)',  text: '#D6A35C' },
  BOS:   { bg: 'rgba(63,182,139,0.15)',  text: '#3FB68B' },
  FVG:   { bg: 'rgba(99,155,235,0.15)',  text: '#639BEB' },
  OB:    { bg: 'rgba(167,139,250,0.15)', text: '#A78BFA' },
};

function signalColor(sig: string) {
  return SIGNAL_COLORS[sig] ?? { bg: 'rgba(255,255,255,0.08)', text: colors.text };
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function fmtPrice(v: number): string {
  if (v >= 1000) return v.toFixed(2);
  if (v >= 1)    return v.toFixed(4);
  return v.toFixed(5);
}

function fmtTime(iso: string): string {
  try {
    const d = new Date(iso);
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const mins = Math.floor(diffMs / 60000);
    if (mins < 1)   return 'just now';
    if (mins < 60)  return `${mins}m ago`;
    return `${Math.floor(mins / 60)}h ago`;
  } catch { return ''; }
}

// ─── Chip component ─────────────────────────────────────────────────────────

function Chip({
  label, active, onPress,
}: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      style={[styles.chip, active && styles.chipActive]}
    >
      <Text style={[styles.chipText, active && styles.chipTextActive]}>{label}</Text>
    </Pressable>
  );
}

// ─── Signal card ────────────────────────────────────────────────────────────

const SignalCard = React.memo(function SignalCard({
  item,
  onBuild,
}: {
  item: ScannerSignal;
  onBuild: (item: ScannerSignal) => void;
}) {
  const sc = signalColor(item.signal);
  const isLong = item.direction === 'LONG';

  return (
    <View style={styles.card}>
      {/* Left direction stripe */}
      <View style={[styles.stripe, { backgroundColor: isLong ? colors.positive : colors.negative }]} />

      <View style={{ flex: 1, padding: spacing.md }}>
        {/* Row 1: pair + signal badge + direction */}
        <View style={styles.cardRow}>
          <Text style={styles.pair}>{item.pair}</Text>

          <View style={[styles.sigBadge, { backgroundColor: sc.bg }]}>
            <Text style={[styles.sigBadgeText, { color: sc.text }]}>{item.signal}</Text>
          </View>

          <View style={[styles.dirBadge, { backgroundColor: isLong ? colors.positiveDim : colors.negativeDim }]}>
            <Ionicons
              name={isLong ? 'arrow-up' : 'arrow-down'}
              size={10}
              color={isLong ? colors.positive : colors.negative}
            />
            <Text style={[styles.dirText, { color: isLong ? colors.positive : colors.negative }]}>
              {item.direction}
            </Text>
          </View>

          <View style={styles.tfBadge}>
            <Text style={styles.tfText}>{item.timeframe}</Text>
          </View>

          <View style={{ flex: 1 }} />
          <Text style={styles.price}>{fmtPrice(item.price)}</Text>
        </View>

        {/* Row 2: description */}
        <Text style={styles.desc} numberOfLines={1}>{item.desc}</Text>

        {/* Row 3: build button */}
        <Pressable
          onPress={() => onBuild(item)}
          style={({ pressed }) => [styles.buildBtn, pressed && { opacity: 0.75 }]}
        >
          <Ionicons name="add-circle-outline" size={13} color={colors.accent} />
          <Text style={styles.buildBtnText}>Build strategy</Text>
        </Pressable>
      </View>
    </View>
  );
});

// ─── Main screen ────────────────────────────────────────────────────────────

export default function ForexScannerScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router  = useRouter();

  const [sigFilter, setSigFilter] = useState<SignalFilter>('ALL');
  const [tfFilter,  setTfFilter]  = useState<TfFilter>('ALL');
  const [dirFilter, setDirFilter] = useState<DirFilter>('ALL');

  const { data, isLoading, isFetching, isError, refetch, dataUpdatedAt } =
    useQuery<ScannerResponse>({
      queryKey: ['forex-scanner', uid],
      queryFn:  () => apiGet<ScannerResponse>(`/api/forex/scanner?uid=${uid}`),
      staleTime: 55_000,
      refetchInterval: 60_000,
      enabled: !!uid,
    });

  const filtered = (data?.signals ?? []).filter(s => {
    if (sigFilter !== 'ALL' && s.signal !== sigFilter) return false;
    if (tfFilter  !== 'ALL' && s.timeframe !== tfFilter) return false;
    if (dirFilter !== 'ALL' && s.direction !== dirFilter) return false;
    return true;
  });

  const onBuild = useCallback((item: ScannerSignal) => {
    Haptics.selectionAsync();
    router.push('/wizard' as any);
  }, [router]);

  const onRefresh = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    refetch();
  }, [refetch]);

  const scannedAgo = data?.scanned_at ? fmtTime(data.scanned_at) : '';

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>

      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} style={styles.backBtn} hitSlop={12}>
          <Ionicons name="chevron-back" size={22} color={colors.text} />
        </Pressable>
        <View style={{ flex: 1 }}>
          <Text style={styles.title}>Forex Scanner</Text>
          <Text style={styles.subtitle}>
            {data
              ? `${data.pairs_scanned} pairs · ${data.count} signals · ${scannedAgo}`
              : 'Live market structure signals'}
          </Text>
        </View>
        <Pressable
          onPress={onRefresh}
          style={styles.refreshBtn}
          hitSlop={8}
        >
          {isFetching
            ? <ActivityIndicator size="small" color={colors.accent} />
            : <Ionicons name="refresh" size={18} color={colors.accent} />}
        </Pressable>
      </View>

      {/* Filter chips */}
      <View style={styles.filtersWrap}>
        <View style={styles.filterRow}>
          {(['ALL', 'CHoCH', 'BOS', 'FVG', 'OB'] as SignalFilter[]).map(f => (
            <Chip key={f} label={f} active={sigFilter === f} onPress={() => setSigFilter(f)} />
          ))}
        </View>
        <View style={styles.filterRow}>
          {(['ALL', '15m', '1h'] as TfFilter[]).map(f => (
            <Chip key={f} label={f} active={tfFilter === f} onPress={() => setTfFilter(f)} />
          ))}
          <View style={{ width: spacing.lg }} />
          {(['ALL', 'LONG', 'SHORT'] as DirFilter[]).map(f => (
            <Chip
              key={f}
              label={f}
              active={dirFilter === f}
              onPress={() => setDirFilter(f)}
            />
          ))}
        </View>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        {Object.entries(SIGNAL_COLORS).map(([sig, sc]) => (
          <View key={sig} style={styles.legendItem}>
            <View style={[styles.legendDot, { backgroundColor: sc.text }]} />
            <Text style={[styles.legendText, { color: sc.text }]}>{sig}</Text>
          </View>
        ))}
      </View>

      {/* Content */}
      {isLoading ? (
        <View style={styles.center}>
          <ActivityIndicator size="large" color={colors.accent} />
          <Text style={styles.loadingText}>Scanning 11 pairs…</Text>
        </View>
      ) : isError ? (
        <View style={styles.center}>
          <Ionicons name="cloud-offline-outline" size={44} color={colors.textMute} />
          <Text style={styles.emptyTitle}>Scan failed</Text>
          <Text style={styles.emptyHint}>Pull down to retry.</Text>
          <Pressable onPress={onRefresh} style={styles.retryBtn}>
            <Text style={styles.retryText}>Retry</Text>
          </Pressable>
        </View>
      ) : filtered.length === 0 ? (
        <View style={styles.center}>
          <Ionicons name="search-outline" size={44} color={colors.textMute} />
          <Text style={styles.emptyTitle}>No signals right now</Text>
          <Text style={styles.emptyHint}>
            {data?.count === 0
              ? 'No active BOS/CHoCH/FVG/OB patterns detected across all pairs. Markets may be ranging — check back later.'
              : 'No signals match your current filters.'}
          </Text>
        </View>
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(item, idx) => `${item.pair}-${item.timeframe}-${item.sig_key}-${idx}`}
          renderItem={({ item }) => <SignalCard item={item} onBuild={onBuild} />}
          ItemSeparatorComponent={() => <View style={{ height: spacing.sm }} />}
          contentContainerStyle={styles.list}
          refreshControl={
            <RefreshControl
              refreshing={isFetching && !isLoading}
              onRefresh={onRefresh}
              tintColor={colors.accent}
              colors={[colors.accent]}
            />
          }
          ListFooterComponent={
            <Text style={styles.footer}>
              Auto-refreshes every 60s · Cached results shown if market data is slow
            </Text>
          }
        />
      )}
    </View>
  );
}

// ─── Styles ─────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
    gap: spacing.sm,
  },
  backBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.md,
    backgroundColor: colors.card,
  },
  title: {
    fontFamily: font.bold,
    fontSize: 18,
    color: colors.text,
    letterSpacing: -0.3,
  },
  subtitle: {
    fontFamily: font.regular,
    fontSize: 12,
    color: colors.textMute,
    marginTop: 1,
  },
  refreshBtn: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: radius.md,
    backgroundColor: colors.card,
  },
  filtersWrap: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.sm,
    gap: spacing.xs,
  },
  filterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.xs,
    marginBottom: 2,
  },
  chip: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.sm,
    backgroundColor: colors.card,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  chipActive: {
    backgroundColor: colors.accent,
    borderColor: colors.accent,
  },
  chipText: {
    fontFamily: font.medium,
    fontSize: 12,
    color: colors.textDim,
  },
  chipTextActive: {
    color: colors.accentText,
  },
  legend: {
    flexDirection: 'row',
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: spacing.md,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: colors.border,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: {
    width: 6,
    height: 6,
    borderRadius: 3,
  },
  legendText: {
    fontFamily: font.medium,
    fontSize: 11,
  },
  list: {
    paddingHorizontal: spacing.md,
    paddingTop: spacing.md,
    paddingBottom: 40,
  },
  card: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    overflow: 'hidden',
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  stripe: {
    width: 3,
  },
  cardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 6,
  },
  pair: {
    fontFamily: font.bold,
    fontSize: 15,
    color: colors.text,
    letterSpacing: -0.2,
  },
  sigBadge: {
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: radius.sm,
  },
  sigBadgeText: {
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.4,
  },
  dirBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.sm,
  },
  dirText: {
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.4,
  },
  tfBadge: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: radius.sm,
    backgroundColor: colors.cardHi,
  },
  tfText: {
    fontFamily: font.medium,
    fontSize: 11,
    color: colors.textDim,
  },
  price: {
    fontFamily: font.bold,
    fontSize: 13,
    color: colors.textDim,
    fontVariant: ['tabular-nums'],
  },
  desc: {
    fontFamily: font.regular,
    fontSize: 12,
    color: colors.textMute,
    marginBottom: 10,
  },
  buildBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    alignSelf: 'flex-start',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.sm,
    backgroundColor: colors.cardHi,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.borderHi,
  },
  buildBtnText: {
    fontFamily: font.medium,
    fontSize: 12,
    color: colors.accent,
  },
  center: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: spacing.xl,
    gap: spacing.sm,
  },
  loadingText: {
    fontFamily: font.regular,
    fontSize: 14,
    color: colors.textMute,
    marginTop: spacing.sm,
  },
  emptyTitle: {
    fontFamily: font.bold,
    fontSize: 17,
    color: colors.text,
    letterSpacing: -0.3,
    marginTop: spacing.sm,
    textAlign: 'center',
  },
  emptyHint: {
    fontFamily: font.regular,
    fontSize: 13,
    color: colors.textMute,
    textAlign: 'center',
    lineHeight: 20,
  },
  retryBtn: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: radius.md,
    backgroundColor: colors.card,
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: colors.border,
  },
  retryText: {
    fontFamily: font.medium,
    fontSize: 14,
    color: colors.text,
  },
  footer: {
    fontFamily: font.regular,
    fontSize: 11,
    color: colors.textMute,
    textAlign: 'center',
    marginTop: spacing.lg,
  },
});
