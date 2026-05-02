import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  ActivityIndicator,
  RefreshControl,
  Pressable,
  ScrollView,
} from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, RadialGradient, Stop, Rect, Circle } from 'react-native-svg';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { AmbientBg } from '@/components/AmbientBg';
import { Sparkline } from '@/components/Sparkline';
import { MiniDonut } from '@/components/MiniDonut';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type MarketplaceListing } from '@/lib/api';

type SortKey = 'top' | 'trending' | 'new' | 'price';
const SORTS: Array<{ key: SortKey; label: string; icon: keyof typeof Ionicons.glyphMap }> = [
  { key: 'top',      label: 'Top',      icon: 'trophy-outline' },
  { key: 'trending', label: 'Trending', icon: 'flame-outline' },
  { key: 'new',      label: 'New',      icon: 'sparkles-outline' },
  { key: 'price',    label: 'Free',     icon: 'gift-outline' },
];

function fmtPnl(v: number | null): string {
  if (v === null || v === undefined) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(1)}%`;
}

function avatarPalette(name: string): [string, string] {
  const palettes: Array<[string, string]> = [
    ['#a78bfa', '#7c3aed'],
    ['#22d3ee', '#3b82f6'],
    ['#34d399', '#10b981'],
    ['#fbbf24', '#f59e0b'],
    ['#f472b6', '#db2777'],
    ['#5eead4', '#22d3ee'],
    ['#f5b754', '#d97706'],
  ];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return palettes[h % palettes.length];
}

const ListingCard = React.memo(function ListingCard({ m, onPress }: { m: MarketplaceListing; onPress: () => void }) {
  const showLive = m.live_pnl !== null && m.live_trades >= 3;
  const headlinePnl = showLive ? m.live_pnl! : (m.verified_pnl ?? 0);
  const pnlColor = headlinePnl > 0 ? colors.positive : headlinePnl < 0 ? colors.negative : colors.text;
  const wr = showLive ? (m.live_win_rate ?? 0) : (m.verified_win_rate ?? 0);
  const initial = (m.author_name || '?').charAt(0).toUpperCase();

  const uid = React.useId().replace(/:/g, '');
  const sheenId = `mk-sheen-${uid}`;
  const haloId = `mk-halo-${uid}`;
  const avatarId = `mk-av-${uid}`;
  const [c0, c1] = avatarPalette(m.author_name || '?');
  const tonePrimary = headlinePnl > 0 ? '#34d399' : headlinePnl < 0 ? '#f87171' : '#67e8f9';

  const isFree = m.pricing_model === 'free';

  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.card,
        glow.card,
        pressed && { opacity: 0.92, transform: [{ scale: 0.995 }] },
      ]}
    >
      <Svg style={StyleSheet.absoluteFill} preserveAspectRatio="none" viewBox="0 0 100 100">
        <Defs>
          <SvgLinearGradient id={sheenId} x1="0" y1="0" x2="0" y2="1">
            <Stop offset="0" stopColor="#1c2440" />
            <Stop offset="1" stopColor="#0f1524" />
          </SvgLinearGradient>
          <RadialGradient id={haloId} cx="100%" cy="0%" rx="65%" ry="60%">
            <Stop offset="0" stopColor={tonePrimary} stopOpacity="0.18" />
            <Stop offset="1" stopColor={tonePrimary} stopOpacity="0" />
          </RadialGradient>
        </Defs>
        <Rect width="100" height="100" fill={`url(#${sheenId})`} />
        <Rect width="100" height="100" fill={`url(#${haloId})`} />
      </Svg>

      {m.is_featured ? (
        <View style={styles.ribbon}>
          <Ionicons name="star" size={10} color="#0a1024" />
          <Text style={styles.ribbonText}>FEATURED</Text>
        </View>
      ) : null}

      <View style={styles.cardInner}>
        <View style={styles.headerRow}>
          <View style={styles.avatarWrap}>
            <Svg width={40} height={40}>
              <Defs>
                <SvgLinearGradient id={avatarId} x1="0" y1="0" x2="1" y2="1">
                  <Stop offset="0" stopColor={c0} />
                  <Stop offset="1" stopColor={c1} />
                </SvgLinearGradient>
              </Defs>
              <Circle cx={20} cy={20} r={19.5} fill={`url(#${avatarId})`} />
            </Svg>
            <View style={styles.avatarLetter} pointerEvents="none">
              <Text style={styles.avatarLetterText}>{initial}</Text>
            </View>
          </View>

          <View style={{ flex: 1, marginLeft: spacing.md }}>
            <Text style={styles.title} numberOfLines={1}>{m.title}</Text>
            <View style={styles.authorRow}>
              <Text style={styles.author} numberOfLines={1}>by {m.author_name}</Text>
              {m.is_verified ? (
                <Ionicons name="checkmark-circle" size={12} color={colors.accent} />
              ) : null}
            </View>
          </View>

          <View
            style={[
              styles.priceBox,
              isFree ? styles.priceBoxFree : styles.priceBoxPaid,
            ]}
          >
            <Text style={isFree ? styles.priceFreeText : styles.pricePaidText}>
              {isFree ? 'FREE' : `$${m.price_usdt}`}
            </Text>
          </View>
        </View>

        {m.summary ? (
          <Text style={styles.summary} numberOfLines={2}>{m.summary}</Text>
        ) : null}

        {/* Hero performance + sparkline */}
        <View style={styles.heroRow}>
          <View style={styles.heroLeft}>
            <Text style={styles.heroLabel}>{showLive ? 'LIVE P&L' : 'AUTHOR P&L'}</Text>
            <Text style={[styles.heroValue, { color: pnlColor }]}>
              {fmtPnl(headlinePnl)}
            </Text>
            {showLive ? (
              <Text style={styles.heroSub}>{m.live_trades} live trades</Text>
            ) : (
              <Text style={styles.heroSub}>Verified backtest</Text>
            )}
          </View>
          <View style={styles.heroRight}>
            {m.equity_curve && m.equity_curve.length >= 2 ? (
              <Sparkline values={m.equity_curve} width={140} height={56} strokeWidth={2} />
            ) : (
              <View style={[styles.sparkPlaceholder, { width: 140, height: 56 }]}>
                <Text style={styles.sparkPlaceholderText}>No equity curve yet</Text>
              </View>
            )}
          </View>
        </View>

        {/* Footer metrics */}
        <View style={styles.metaRow}>
          <View style={styles.metaCell}>
            <MiniDonut value={wr} size={28} stroke={3.5} label=" " />
            <View>
              <Text style={styles.metaLabel}>Win</Text>
              <Text style={styles.metaValue}>
                {(showLive ? m.live_trades >= 3 : (m.verified_pnl !== null)) ? `${wr.toFixed(0)}%` : '—'}
              </Text>
            </View>
          </View>
          <View style={styles.metaCell}>
            <View style={styles.metaIcon}>
              <Ionicons name="git-network-outline" size={14} color={colors.violet} />
            </View>
            <View>
              <Text style={styles.metaLabel}>Clones</Text>
              <Text style={styles.metaValue}>{m.clone_count}</Text>
            </View>
          </View>
          <View style={styles.metaCell}>
            <View style={[styles.metaIcon, { backgroundColor: colors.warningDim, borderColor: 'rgba(251,191,36,0.32)' }]}>
              <Ionicons name="star" size={13} color={colors.warning} />
            </View>
            <View>
              <Text style={styles.metaLabel}>Rating</Text>
              <Text style={styles.metaValue}>
                {m.rating_count > 0 ? m.avg_rating.toFixed(1) : '—'}
              </Text>
            </View>
          </View>
        </View>

        {/* Badges row */}
        <View style={styles.badges}>
          <Pill label={m.category} tone="neutral" small />
          {m.is_trending ? <Pill label="🔥 Trending" tone="warning" small /> : null}
          {m.is_verified && !m.is_featured ? <Pill label="✓ Verified" tone="accent" small /> : null}
        </View>
      </View>
    </Pressable>
  );
});

function SortChips({ value, onChange }: { value: SortKey; onChange: (s: SortKey) => void }) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.chipRow}
    >
      {SORTS.map((s) => {
        const active = s.key === value;
        return (
          <Pressable
            key={s.key}
            onPress={() => {
              if (active) return;
              Haptics.selectionAsync().catch(() => {});
              onChange(s.key);
            }}
            style={({ pressed }) => [
              styles.chip,
              active && styles.chipActive,
              pressed && { opacity: 0.85 },
            ]}
          >
            <Ionicons
              name={s.icon}
              size={13}
              color={active ? colors.accentText : colors.textDim}
            />
            <Text style={[styles.chipText, active && styles.chipTextActive]}>{s.label}</Text>
          </Pressable>
        );
      })}
    </ScrollView>
  );
}

export default function MarketplaceScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const [sort, setSort] = useState<SortKey>('top');

  const apiSort = sort === 'price' ? 'top' : sort;

  const { data, isLoading, isFetching, refetch, isError } = useQuery({
    queryKey: ['marketplace', uid, apiSort],
    queryFn: () => apiGet<MarketplaceListing[]>('/api/marketplace', uid, { sort: apiSort }),
    enabled: !!uid,
  });

  const onRefresh = useCallback(() => { refetch(); }, [refetch]);

  const filtered = useMemo(() => {
    const list = data || [];
    if (sort === 'price') return list.filter((m) => m.pricing_model === 'free');
    return list;
  }, [data, sort]);

  const Header = (
    <View style={styles.header}>
      <View style={{ flexDirection: 'row', alignItems: 'center', gap: 10 }}>
        <View style={styles.headerIconWrap}>
          <Ionicons name="storefront" size={20} color={colors.accent} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={styles.titleHero}>Market</Text>
          <Text style={styles.subtitleHero}>
            {data?.length
              ? `${data.length} community strateg${data.length === 1 ? 'y' : 'ies'} · ranked by performance`
              : 'Top community strategies, ranked by performance'}
          </Text>
        </View>
      </View>
      <SortChips value={sort} onChange={setSort} />
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
          <EmptyState icon="cloud-offline-outline" title="Couldn't load marketplace" hint="Pull down to retry." />
        </ScrollView>
      </View>
    );
  }

  return (
    <View style={[styles.root, { paddingTop: insets.top }]}>
      <AmbientBg variant="duo" />
      <FlatList
        data={filtered}
        keyExtractor={(m) => `m-${m.id}`}
        renderItem={({ item }) => (
          <ListingCard m={item} onPress={() => router.push(`/listing/${item.id}` as any)} />
        )}
        ListHeaderComponent={Header}
        ItemSeparatorComponent={() => <View style={{ height: spacing.md }} />}
        contentContainerStyle={styles.listContent}
        initialNumToRender={5}
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
          <EmptyState
            icon={sort === 'price' ? 'gift-outline' : 'storefront-outline'}
            title={sort === 'price' ? 'No free listings yet' : 'Marketplace is quiet'}
            hint={sort === 'price' ? 'Switch to Top to browse paid strategies.' : 'Be the first to publish a strategy from the web portal.'}
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
  header: { paddingHorizontal: spacing.lg, paddingTop: spacing.md, paddingBottom: spacing.sm },
  headerIconWrap: {
    width: 36, height: 36, borderRadius: 10,
    backgroundColor: colors.accentDim,
    borderWidth: 1, borderColor: 'rgba(34,211,238,0.32)',
    alignItems: 'center', justifyContent: 'center',
  },
  titleHero: { color: colors.text, fontFamily: font.black, fontSize: 30, letterSpacing: -1.0 },
  subtitleHero: { color: colors.textDim, fontFamily: font.regular, fontSize: 13.5, marginTop: 2, lineHeight: 18 },
  listContent: { paddingHorizontal: spacing.lg, paddingBottom: spacing.xxl + 96 },

  // Card
  card: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    overflow: 'hidden',
  },
  cardInner: { padding: spacing.lg },
  ribbon: {
    position: 'absolute',
    top: 12,
    right: -28,
    backgroundColor: colors.gold,
    paddingHorizontal: 32,
    paddingVertical: 3,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    transform: [{ rotate: '32deg' }],
    zIndex: 5,
  },
  ribbonText: {
    color: '#0a1024',
    fontFamily: font.black,
    fontSize: 9,
    letterSpacing: 1.0,
  },
  headerRow: { flexDirection: 'row', alignItems: 'center', gap: 4 },
  avatarWrap: {
    width: 40,
    height: 40,
    position: 'relative',
  },
  avatarLetter: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
  },
  avatarLetterText: {
    color: '#fff',
    fontFamily: font.black,
    fontSize: 16,
    textShadowColor: 'rgba(0,0,0,0.3)',
    textShadowOffset: { width: 0, height: 1 },
    textShadowRadius: 2,
  },
  title: { color: colors.text, fontFamily: font.black, fontSize: 16.5, letterSpacing: -0.4 },
  authorRow: { flexDirection: 'row', alignItems: 'center', gap: 4, marginTop: 2 },
  author: { color: colors.textDim, fontFamily: font.semibold, fontSize: 12 },
  priceBox: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  priceBoxFree: {
    backgroundColor: colors.positiveDim,
    borderColor: 'rgba(52,211,153,0.42)',
  },
  priceBoxPaid: {
    backgroundColor: colors.accentDim,
    borderColor: 'rgba(34,211,238,0.36)',
  },
  priceFreeText: {
    color: colors.positive,
    fontFamily: font.black,
    fontSize: 11,
    letterSpacing: 0.7,
  },
  pricePaidText: {
    color: colors.accent,
    fontFamily: font.black,
    fontSize: 13,
    fontVariant: ['tabular-nums'],
  },
  summary: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    marginTop: spacing.md,
    lineHeight: 18,
  },

  heroRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
    gap: spacing.md,
  },
  heroLeft: { flex: 1 },
  heroRight: { alignItems: 'flex-end' },
  heroLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
  },
  heroValue: {
    fontFamily: font.black,
    fontSize: 28,
    letterSpacing: -1.0,
    fontVariant: ['tabular-nums'],
    marginTop: 2,
  },
  heroSub: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
    marginTop: 2,
  },
  sparkPlaceholder: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sparkPlaceholderText: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 10,
  },

  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    gap: spacing.lg,
  },
  metaCell: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  metaIcon: {
    width: 28,
    height: 28,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.violetDim,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.32)',
  },
  metaLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  metaValue: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 13,
    fontVariant: ['tabular-nums'],
    marginTop: 1,
  },
  badges: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
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
});
