import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  Pressable,
  useWindowDimensions,
  ActivityIndicator,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useQuery } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { CandleChart, type ChartZone } from '@/components/CandleChart';
import { CoinChip } from '@/components/CoinChip';
import { SectionLabel } from '@/components/SectionLabel';
import { Pill } from '@/components/Pill';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import {
  apiGet,
  type TradeCandlesResponse,
  type TradeTicker,
  type TradeFvgResponse,
  type TradeWallReport,
  type TradeWall,
} from '@/lib/api';

// Curated subset of the backend whitelist — picks the highest-volume pairs
// that traders actually want a chart for. Order = display order in the picker.
const TRADE_SYMBOLS: { sym: string; label: string }[] = [
  { sym: 'BTC',   label: 'Bitcoin' },
  { sym: 'ETH',   label: 'Ethereum' },
  { sym: 'SOL',   label: 'Solana' },
  { sym: 'BNB',   label: 'BNB' },
  { sym: 'XRP',   label: 'XRP' },
  { sym: 'DOGE',  label: 'Dogecoin' },
  { sym: 'ADA',   label: 'Cardano' },
  { sym: 'AVAX',  label: 'Avalanche' },
  { sym: 'LINK',  label: 'Chainlink' },
  { sym: 'TON',   label: 'Toncoin' },
  { sym: 'DOT',   label: 'Polkadot' },
  { sym: 'MATIC', label: 'Polygon' },
  { sym: 'NEAR',  label: 'Near' },
  { sym: 'LTC',   label: 'Litecoin' },
  { sym: 'ARB',   label: 'Arbitrum' },
  { sym: 'OP',    label: 'Optimism' },
  { sym: 'SUI',   label: 'Sui' },
  { sym: 'INJ',   label: 'Injective' },
  { sym: 'APT',   label: 'Aptos' },
  { sym: 'PEPE',  label: 'Pepe' },
  { sym: 'WIF',   label: 'dogwifhat' },
  { sym: 'JUP',   label: 'Jupiter' },
];

const TIMEFRAMES: { tf: string; label: string }[] = [
  { tf: '1m',  label: '1m'  },
  { tf: '5m',  label: '5m'  },
  { tf: '15m', label: '15m' },
  { tf: '1h',  label: '1h'  },
];

// Compact USD price formatter — keeps 4 decimals on micro-cap tokens but
// switches to whole-dollar grouping for BTC/ETH-class assets.
function fmtPrice(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '—';
  if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (p >= 1)    return p.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (p >= 0.01) return p.toLocaleString(undefined, { maximumFractionDigits: 5 });
  return p.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

function fmtUsd(n: number): string {
  const abs = Math.abs(n);
  if (abs >= 1_000_000_000) return `$${(n / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000)     return `$${(n / 1_000_000).toFixed(2)}M`;
  if (abs >= 1_000)         return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

function pressureColor(score: number): string {
  if (score >  0.30) return colors.positive;
  if (score >  0.05) return '#7dd3a8';
  if (score < -0.30) return colors.negative;
  if (score < -0.05) return '#f3a3a3';
  return colors.textDim;
}

export default function TradeScreen() {
  const [symbol, setSymbol] = useState<string>('BTC');
  const [tf, setTf]         = useState<string>('5m');
  const [showFvg, setShowFvg] = useState<boolean>(true);

  // ─── Data fetches (react-query polling) ──────────────────────────────────
  // Candles refresh every 6s — short enough to feel live without spamming.
  const candlesQ = useQuery({
    queryKey: ['trade-candles', symbol, tf],
    queryFn: () =>
      apiGet<TradeCandlesResponse>(`/api/trade/candles/${symbol}`, undefined, { tf, limit: 120 }),
    refetchInterval: 6_000,
    staleTime: 4_000,
  });

  const tickerQ = useQuery({
    queryKey: ['trade-ticker', symbol],
    queryFn: () => apiGet<TradeTicker>(`/api/trade/ticker/${symbol}`),
    refetchInterval: 4_000,
    staleTime: 2_000,
  });

  const fvgQ = useQuery({
    queryKey: ['trade-fvg', symbol, tf],
    queryFn: () => apiGet<TradeFvgResponse>(`/api/trade/fvg/${symbol}`, undefined, { tf, limit: 20 }),
    refetchInterval: 12_000,
    staleTime: 8_000,
    enabled: showFvg,
  });

  const wallsQ = useQuery({
    queryKey: ['trade-walls', symbol],
    queryFn: () => apiGet<TradeWallReport>(`/api/trade/walls/${symbol}`),
    refetchInterval: 15_000,
    staleTime: 10_000,
  });

  const onRefresh = useCallback(() => {
    candlesQ.refetch();
    tickerQ.refetch();
    if (showFvg) fvgQ.refetch();
    wallsQ.refetch();
  }, [candlesQ, tickerQ, fvgQ, wallsQ, showFvg]);

  const isRefreshing =
    candlesQ.isFetching || tickerQ.isFetching || wallsQ.isFetching || (showFvg && fvgQ.isFetching);

  // ─── Chart data ──────────────────────────────────────────────────────────
  const chartCandles = candlesQ.data?.candles || [];
  // useWindowDimensions is reactive — recomputes on device rotation, split-view
  // changes, and external display attach so the chart resizes correctly.
  const { width: screenW } = useWindowDimensions();
  const chartW = Math.max(screenW - spacing.lg * 2 - 2, 200);

  // Map FVG zones into ChartZone format. We only show gaps whose formation
  // candle falls within the visible chart window — anything older would draw
  // off-canvas and just clutter the legend count.
  const chartZones = useMemo<ChartZone[]>(() => {
    if (!showFvg) return [];
    const gaps = fvgQ.data?.gaps || [];
    if (chartCandles.length === 0 || gaps.length === 0) return [];
    const minTime = chartCandles[0].time;
    return gaps
      .filter((g) => g.time >= minTime)
      .map((g) => ({
        fromTime: g.time,
        toTime:   g.filled_at || undefined,  // unfilled → extends to chart edge
        top:      g.top,
        bottom:   g.bottom,
        side:     g.side,
        dim:      g.filled,
      }));
  }, [showFvg, fvgQ.data, chartCandles]);

  const fvgCounts = useMemo(() => {
    const gaps = fvgQ.data?.gaps || [];
    const unfilled = gaps.filter((g) => !g.filled);
    return {
      bull: unfilled.filter((g) => g.side === 'bull').length,
      bear: unfilled.filter((g) => g.side === 'bear').length,
      total: unfilled.length,
    };
  }, [fvgQ.data]);

  const ticker = tickerQ.data;
  const walls = wallsQ.data;
  const livePrice = ticker?.price || (chartCandles.length ? chartCandles[chartCandles.length - 1].close : 0);

  const onPickSymbol = useCallback((next: string) => {
    if (next === symbol) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setSymbol(next);
  }, [symbol]);

  const onPickTf = useCallback((next: string) => {
    if (next === tf) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setTf(next);
  }, [tf]);

  const onToggleFvg = useCallback(() => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setShowFvg((v) => !v);
  }, []);

  return (
    <Screen
      title="Trade"
      subtitle="Live charts, FVG zones, and order-book walls."
      ambient="cyan"
      refreshing={isRefreshing && !candlesQ.isLoading}
      onRefresh={onRefresh}
    >
      {/* ─── Coin picker (horizontal scroll) ──────────────────────────── */}
      <View style={styles.coinPickerWrap}>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.coinPickerContent}
        >
          {TRADE_SYMBOLS.map(({ sym, label }) => {
            const active = sym === symbol;
            return (
              <Pressable
                key={sym}
                onPress={() => onPickSymbol(sym)}
                style={({ pressed }) => [
                  styles.coinPill,
                  active && styles.coinPillActive,
                  pressed && !active && { opacity: 0.85 },
                ]}
              >
                <CoinChip symbol={sym} size={26} />
                <View style={{ marginLeft: 8 }}>
                  <Text style={[styles.coinPillSym, active && { color: colors.accent }]}>{sym}</Text>
                  <Text style={styles.coinPillLabel}>{label}</Text>
                </View>
              </Pressable>
            );
          })}
        </ScrollView>
      </View>

      {/* ─── Ticker bar ──────────────────────────────────────────────── */}
      <View style={styles.tickerCard}>
        <View style={styles.tickerTopRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.tickerSymbol}>{symbol}/USDT</Text>
            <Text style={styles.tickerPrice}>${fmtPrice(livePrice)}</Text>
          </View>
          <View style={styles.tickerRight}>
            {ticker ? (
              <>
                <Text
                  style={[
                    styles.tickerChange,
                    { color: ticker.change_pct >= 0 ? colors.positive : colors.negative },
                  ]}
                >
                  {ticker.change_pct >= 0 ? '+' : ''}{ticker.change_pct.toFixed(2)}%
                </Text>
                <Text style={styles.tickerChangeAbs}>
                  {ticker.change_abs >= 0 ? '+' : ''}${fmtPrice(Math.abs(ticker.change_abs))}
                </Text>
              </>
            ) : (
              <Text style={styles.tickerChangeAbs}>—</Text>
            )}
          </View>
        </View>

        {ticker ? (
          <View style={styles.tickerStatsRow}>
            <TickerStat label="24H HIGH"   value={`$${fmtPrice(ticker.high_24h)}`} />
            <TickerStat label="24H LOW"    value={`$${fmtPrice(ticker.low_24h)}`} />
            <TickerStat label="24H VOL"    value={fmtUsd(ticker.vol_24h_quote)} />
          </View>
        ) : null}
      </View>

      {/* ─── Timeframe + FVG toggle ────────────────────────────────────── */}
      <View style={styles.tfRow}>
        <View style={styles.tfGroup}>
          {TIMEFRAMES.map(({ tf: t, label }) => {
            const active = t === tf;
            return (
              <Pressable
                key={t}
                onPress={() => onPickTf(t)}
                style={({ pressed }) => [
                  styles.tfBtn,
                  active && styles.tfBtnActive,
                  pressed && !active && { opacity: 0.85 },
                ]}
              >
                <Text style={[styles.tfBtnText, active && { color: colors.accentText }]}>{label}</Text>
              </Pressable>
            );
          })}
        </View>

        <Pressable
          onPress={onToggleFvg}
          style={({ pressed }) => [
            styles.fvgToggle,
            showFvg && styles.fvgToggleActive,
            pressed && { opacity: 0.85 },
          ]}
        >
          <Ionicons
            name={showFvg ? 'eye' : 'eye-off-outline'}
            size={14}
            color={showFvg ? colors.accent : colors.textMute}
          />
          <Text style={[styles.fvgToggleText, showFvg && { color: colors.accent }]}>
            FVG {showFvg ? 'ON' : 'OFF'}
          </Text>
          {showFvg && fvgCounts.total > 0 ? (
            <View style={styles.fvgCountChip}>
              <Text style={styles.fvgCountChipText}>
                <Text style={{ color: colors.positive }}>{fvgCounts.bull}</Text>
                <Text style={{ color: colors.textMute }}>/</Text>
                <Text style={{ color: colors.negative }}>{fvgCounts.bear}</Text>
              </Text>
            </View>
          ) : null}
        </Pressable>
      </View>

      {/* ─── Chart ──────────────────────────────────────────────────── */}
      <View style={{ marginTop: spacing.md }}>
        {candlesQ.isLoading ? (
          <View style={styles.loadingCard}>
            <ActivityIndicator size="small" color={colors.accent} />
            <Text style={styles.loadingText}>Loading {symbol} candles…</Text>
          </View>
        ) : candlesQ.isError ? (
          <View style={styles.errorCard}>
            <Ionicons name="warning" size={18} color={colors.warning} />
            <Text style={styles.errorText}>
              Couldn't load chart. Pull down to retry.
            </Text>
          </View>
        ) : (
          <CandleChart
            candles={chartCandles}
            zones={chartZones}
            width={chartW}
            height={240}
            symbol={`${symbol}USDT`}
            tf={tf}
          />
        )}
      </View>

      {/* ─── FVG legend ─────────────────────────────────────────────── */}
      {showFvg ? (
        <View style={styles.fvgLegend}>
          <View style={styles.fvgLegendItem}>
            <View style={[styles.fvgSwatch, { backgroundColor: colors.positive }]} />
            <Text style={styles.fvgLegendText}>
              Bull FVG · support
              {fvgQ.data ? ` · ${fvgCounts.bull} active` : ''}
            </Text>
          </View>
          <View style={styles.fvgLegendItem}>
            <View style={[styles.fvgSwatch, { backgroundColor: colors.negative }]} />
            <Text style={styles.fvgLegendText}>
              Bear FVG · resistance
              {fvgQ.data ? ` · ${fvgCounts.bear} active` : ''}
            </Text>
          </View>
          <Text style={styles.fvgHint}>
            ICT 3-bar imbalance gaps · price often retraces to fill them.
          </Text>
        </View>
      ) : null}

      {/* ─── Order-book walls ───────────────────────────────────────── */}
      <View style={{ marginTop: spacing.xl }}>
        <SectionLabel label="Order-book walls" />
      </View>

      {wallsQ.isLoading ? (
        <View style={styles.loadingCard}>
          <ActivityIndicator size="small" color={colors.accent} />
          <Text style={styles.loadingText}>Scanning order books…</Text>
        </View>
      ) : wallsQ.isError || !walls ? (
        <View style={styles.errorCard}>
          <Ionicons name="warning" size={18} color={colors.warning} />
          <Text style={styles.errorText}>
            Order-book scan unavailable. Pull down to retry.
          </Text>
        </View>
      ) : (
        <>
          {/* Pressure summary */}
          <View style={styles.pressureCard}>
            <View style={styles.pressureLeft}>
              <Text style={styles.pressureLabel}>BOOK PRESSURE</Text>
              <Text style={[styles.pressureValue, { color: pressureColor(walls.pressure_score) }]}>
                {walls.pressure_label}
              </Text>
            </View>
            <View style={styles.pressureBar}>
              <View
                style={[
                  styles.pressureFill,
                  {
                    width: `${Math.min(100, Math.abs(walls.pressure_score) * 100)}%`,
                    backgroundColor: pressureColor(walls.pressure_score),
                    alignSelf: walls.pressure_score >= 0 ? 'flex-start' : 'flex-end',
                  },
                ]}
              />
            </View>
          </View>

          {walls.best_zone_to_watch ? (
            <View style={styles.zoneCard}>
              <Ionicons name="flag" size={14} color={colors.accent} />
              <Text style={styles.zoneText}>{walls.best_zone_to_watch}</Text>
            </View>
          ) : null}

          {/* Top buys */}
          <View style={styles.wallsBlock}>
            <Text style={styles.wallsBlockTitle}>
              <Text style={{ color: colors.positive }}>BIDS</Text>
              <Text style={{ color: colors.textMute }}>  · support below price</Text>
            </Text>
            {(walls.top_buys || []).slice(0, 5).map((w, i) => (
              <WallRow key={`buy-${i}`} wall={w} refPrice={walls.price} />
            ))}
            {(!walls.top_buys || walls.top_buys.length === 0) ? (
              <Text style={styles.wallsEmpty}>No major bid walls detected.</Text>
            ) : null}
          </View>

          {/* Top sells */}
          <View style={styles.wallsBlock}>
            <Text style={styles.wallsBlockTitle}>
              <Text style={{ color: colors.negative }}>ASKS</Text>
              <Text style={{ color: colors.textMute }}>  · resistance above price</Text>
            </Text>
            {(walls.top_sells || []).slice(0, 5).map((w, i) => (
              <WallRow key={`sell-${i}`} wall={w} refPrice={walls.price} />
            ))}
            {(!walls.top_sells || walls.top_sells.length === 0) ? (
              <Text style={styles.wallsEmpty}>No major ask walls detected.</Text>
            ) : null}
          </View>

          {walls.ai_summary ? (
            <View style={styles.aiCard}>
              <View style={styles.aiHeader}>
                <Ionicons name="sparkles" size={14} color={colors.violet} />
                <Text style={styles.aiHeaderText}>AI READ</Text>
              </View>
              <Text style={styles.aiBody}>{walls.ai_summary}</Text>
            </View>
          ) : null}

          {walls.exchanges_used && walls.exchanges_used.length ? (
            <View style={styles.exchPills}>
              {walls.exchanges_used.map((e) => (
                <Pill key={e} label={e.toUpperCase()} tone="neutral" small />
              ))}
            </View>
          ) : null}
        </>
      )}

      <View style={{ height: spacing.xxl * 2 }} />
    </Screen>
  );
}

// ─── Sub-components ─────────────────────────────────────────────────────────

function TickerStat({ label, value }: { label: string; value: string }) {
  return (
    <View style={styles.tickerStatCell}>
      <Text style={styles.tickerStatLabel}>{label}</Text>
      <Text style={styles.tickerStatValue}>{value}</Text>
    </View>
  );
}

function WallRow({ wall, refPrice }: { wall: TradeWall; refPrice: number }) {
  const isBuy = wall.side === 'buy';
  const tone = isBuy ? colors.positive : colors.negative;
  const distLabel = `${wall.distance_pct >= 0 ? '+' : ''}${wall.distance_pct.toFixed(2)}%`;
  return (
    <View style={styles.wallRow}>
      <View style={[styles.wallSideBar, { backgroundColor: tone }]} />
      <View style={{ flex: 1 }}>
        <View style={styles.wallTopLine}>
          <Text style={styles.wallPrice}>${fmtPrice(wall.price)}</Text>
          <Text style={[styles.wallDist, { color: tone }]}>{distLabel}</Text>
        </View>
        <View style={styles.wallBottomLine}>
          <Text style={styles.wallSize}>{fmtUsd(wall.size_usd)}</Text>
          {wall.exchanges && wall.exchanges.length ? (
            <Text style={styles.wallExch}>
              {wall.exchanges.length === 1
                ? wall.exchanges[0].toUpperCase()
                : `${wall.exchanges.length} exchanges`}
            </Text>
          ) : null}
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  // Coin picker
  coinPickerWrap: {
    marginHorizontal: -spacing.lg,
    marginTop: -spacing.sm,
    marginBottom: spacing.md,
  },
  coinPickerContent: {
    paddingHorizontal: spacing.lg,
    gap: 8,
  },
  coinPill: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.pill,
    paddingHorizontal: 10,
    paddingVertical: 6,
  },
  coinPillActive: {
    backgroundColor: colors.accentDim,
    borderColor: colors.borderAccent,
  },
  coinPillSym: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 13,
    letterSpacing: 0.3,
  },
  coinPillLabel: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 10,
    marginTop: 1,
  },

  // Ticker
  tickerCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    ...glow.card,
  },
  tickerTopRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  tickerSymbol: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.7,
  },
  tickerPrice: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 28,
    letterSpacing: -0.6,
    fontVariant: ['tabular-nums'],
    marginTop: 2,
  },
  tickerRight: { alignItems: 'flex-end' },
  tickerChange: {
    fontFamily: font.black,
    fontSize: 18,
    fontVariant: ['tabular-nums'],
    letterSpacing: -0.3,
  },
  tickerChangeAbs: {
    color: colors.textMute,
    fontFamily: font.semibold,
    fontSize: 11.5,
    fontVariant: ['tabular-nums'],
    marginTop: 2,
  },
  tickerStatsRow: {
    flexDirection: 'row',
    marginTop: spacing.md,
    paddingTop: spacing.md,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  tickerStatCell: { flex: 1 },
  tickerStatLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.7,
  },
  tickerStatValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 12.5,
    fontVariant: ['tabular-nums'],
    marginTop: 3,
  },

  // TF + FVG row
  tfRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
    gap: spacing.sm,
  },
  tfGroup: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 3,
    flex: 1,
  },
  tfBtn: {
    flex: 1,
    paddingVertical: 7,
    borderRadius: radius.pill,
    alignItems: 'center',
  },
  tfBtnActive: { backgroundColor: colors.accent },
  tfBtnText: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.4,
  },
  fvgToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
  },
  fvgToggleActive: {
    borderColor: 'rgba(34,211,238,0.32)',
    backgroundColor: colors.accentDim,
  },
  fvgToggleText: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.5,
  },
  fvgCountChip: {
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderRadius: radius.pill,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  fvgCountChipText: {
    fontFamily: font.bold,
    fontSize: 11,
    fontVariant: ['tabular-nums'],
  },

  // FVG legend
  fvgLegend: {
    marginTop: spacing.sm,
    backgroundColor: 'rgba(0,0,0,0.28)',
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: colors.border,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    gap: 6,
  },
  fvgLegendItem: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  fvgSwatch: { width: 14, height: 8, borderRadius: 2, opacity: 0.6 },
  fvgLegendText: { color: colors.textDim, fontFamily: font.semibold, fontSize: 11.5 },
  fvgHint: { color: colors.textMute, fontFamily: font.regular, fontSize: 10.5, marginTop: 2 },

  // Loading / error cards
  loadingCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.lg,
    justifyContent: 'center',
  },
  loadingText: { color: colors.textDim, fontFamily: font.semibold, fontSize: 13 },
  errorCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: 'rgba(251,191,36,0.08)',
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(251,191,36,0.28)',
    padding: spacing.md,
  },
  errorText: { color: colors.text, fontFamily: font.semibold, fontSize: 13, flex: 1 },

  // Pressure card
  pressureCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    ...glow.card,
  },
  pressureLeft: { marginBottom: spacing.sm },
  pressureLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
  },
  pressureValue: {
    fontFamily: font.black,
    fontSize: 17,
    letterSpacing: -0.3,
    marginTop: 3,
  },
  pressureBar: {
    height: 6,
    backgroundColor: 'rgba(255,255,255,0.06)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  pressureFill: { height: '100%', borderRadius: 3 },

  // Best-zone callout
  zoneCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    marginTop: spacing.sm,
    backgroundColor: colors.accentDim,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: 'rgba(34,211,238,0.28)',
    padding: spacing.md,
  },
  zoneText: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 12.5,
    flex: 1,
    lineHeight: 17,
  },

  // Walls
  wallsBlock: {
    marginTop: spacing.md,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
  },
  wallsBlockTitle: {
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.7,
    marginBottom: spacing.sm,
  },
  wallsEmpty: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    fontStyle: 'italic',
  },
  wallRow: {
    flexDirection: 'row',
    alignItems: 'stretch',
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.04)',
    gap: 10,
  },
  wallSideBar: { width: 3, borderRadius: 2 },
  wallTopLine: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'baseline',
  },
  wallPrice: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 13.5,
    fontVariant: ['tabular-nums'],
  },
  wallDist: {
    fontFamily: font.bold,
    fontSize: 12,
    fontVariant: ['tabular-nums'],
  },
  wallBottomLine: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 2,
  },
  wallSize: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 11.5,
    fontVariant: ['tabular-nums'],
  },
  wallExch: {
    color: colors.textMute,
    fontFamily: font.semibold,
    fontSize: 10,
    letterSpacing: 0.4,
  },

  // AI summary
  aiCard: {
    marginTop: spacing.md,
    backgroundColor: colors.violetDim,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
    padding: spacing.md,
  },
  aiHeader: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 6 },
  aiHeaderText: {
    color: colors.violet,
    fontFamily: font.black,
    fontSize: 10.5,
    letterSpacing: 0.7,
  },
  aiBody: {
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 18,
  },

  exchPills: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 5,
    marginTop: spacing.md,
    justifyContent: 'center',
  },
});
