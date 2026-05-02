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
import { useMutation, useQuery } from '@tanstack/react-query';

import { Screen } from '@/components/Screen';
import { CandleChart, type ChartZone, type ChartPriceLine } from '@/components/CandleChart';
import { CoinChip } from '@/components/CoinChip';
import { SectionLabel } from '@/components/SectionLabel';
import { Pill } from '@/components/Pill';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import {
  apiGet,
  apiPost,
  type TradeCandlesResponse,
  type TradeTicker,
  type TradeFvgResponse,
  type TradeWallReport,
  type TradeWall,
  type TradeTapeResponse,
  type TradeTapeRow,
  type TradeAiRead,
  type TradeFunding,
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

// ─── Formatters ─────────────────────────────────────────────────────────────

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

function fmtRelMs(tsMs: number): string {
  const diff = Math.max(0, Date.now() - tsMs);
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m`;
  const h = Math.floor(m / 60);
  return `${h}h`;
}

function pressureColor(score: number): string {
  if (score >  0.30) return colors.positive;
  if (score >  0.05) return '#7dd3a8';
  if (score < -0.30) return colors.negative;
  if (score < -0.05) return '#f3a3a3';
  return colors.textDim;
}

// Parse the structured AI Trade Read summary (TRADE: / STOP: / TP1: / TP2:
// / ODDS: / LEVERAGE: / RATIONALE: lines) into a render-friendly object.
type AiPlan = {
  bias?: 'LONG' | 'SHORT' | null;
  entry?: string | null;
  orderType?: string | null;
  stop?: string | null;
  tp1?: string | null;
  tp2?: string | null;
  oddsUp?: number | null;
  oddsDown?: number | null;
  leverage?: string | null;
  rationale: string;
};

function parseAiPlan(raw: string): AiPlan {
  if (!raw) return { rationale: '' };
  const lines = raw.split('\n');
  const plan: AiPlan = { rationale: '' };
  const consumed = new Set<number>();
  const matchOnce = (idx: number, re: RegExp): RegExpMatchArray | null => {
    const m = lines[idx].match(re);
    if (m) consumed.add(idx);
    return m;
  };
  for (let i = 0; i < lines.length; i++) {
    const trade = matchOnce(i, /^\s*TRADE:\s*(LONG|SHORT)\s*@?\s*\$?([0-9][0-9,.]*)?/i);
    if (trade) {
      plan.bias = trade[1].toUpperCase() as 'LONG' | 'SHORT';
      if (trade[2]) plan.entry = trade[2].replace(/,/g, '');
      continue;
    }
    const ot = matchOnce(i, /^\s*ORDER_TYPE:\s*(MARKET|LIMIT)/i);
    if (ot) { plan.orderType = ot[1].toUpperCase(); continue; }
    const stop = matchOnce(i, /^\s*STOP:\s*\$?([0-9][0-9,.]*)/i);
    if (stop) { plan.stop = stop[1].replace(/,/g, ''); continue; }
    const tp1 = matchOnce(i, /^\s*TP1:\s*\$?([0-9][0-9,.]*)/i);
    if (tp1) { plan.tp1 = tp1[1].replace(/,/g, ''); continue; }
    const tp2 = matchOnce(i, /^\s*TP2:\s*\$?([0-9][0-9,.]*)/i);
    if (tp2) { plan.tp2 = tp2[1].replace(/,/g, ''); continue; }
    const odds = matchOnce(i, /^\s*ODDS:\s*UP\s*([0-9]+)\s*%\s*\/\s*DOWN\s*([0-9]+)\s*%/i);
    if (odds) { plan.oddsUp = parseInt(odds[1], 10); plan.oddsDown = parseInt(odds[2], 10); continue; }
    const lev = matchOnce(i, /^\s*LEVERAGE:\s*(.+)$/i);
    if (lev) { plan.leverage = lev[1].trim(); continue; }
  }
  // Everything not consumed becomes the rationale (RATIONALE: line is left in
  // verbatim — we strip the prefix below to keep the text clean).
  plan.rationale = lines
    .map((l, i) => (consumed.has(i) ? null : l))
    .filter((l): l is string => l != null)
    .join('\n')
    .replace(/^\s*RATIONALE:\s*/im, '')
    .trim();
  return plan;
}

export default function TradeScreen() {
  const [symbol, setSymbol] = useState<string>('BTC');
  const [tf, setTf]         = useState<string>('5m');
  const [showFvg, setShowFvg] = useState<boolean>(true);
  const [showWalls, setShowWalls] = useState<boolean>(true);
  const [showTape, setShowTape] = useState<boolean>(true);

  // ─── Data fetches (react-query polling) ──────────────────────────────────
  const candlesQ = useQuery({
    queryKey: ['trade-candles', symbol, tf],
    queryFn: () =>
      apiGet<TradeCandlesResponse>(`/api/trade/candles/${symbol}`, undefined, { tf, limit: 200 }),
    // Backbone candles only need to refresh every few seconds — the visible
    // tick of the trailing candle is handled by the 1s ticker poll which is
    // merged into `liveCandles` below for sub-second-feeling motion.
    refetchInterval: 4_000,
    staleTime: 2_000,
  });

  const tickerQ = useQuery({
    queryKey: ['trade-ticker', symbol],
    queryFn: () => apiGet<TradeTicker>(`/api/trade/ticker/${symbol}`),
    // Poll the lightweight ticker once per second so the chart's trailing
    // candle ticks visibly — the user asked for "by-the-ms" motion and 1s is
    // the sweet spot between perceived liveness and exchange rate-limits.
    refetchInterval: 1_000,
    staleTime: 500,
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

  const tapeQ = useQuery({
    queryKey: ['trade-tape', symbol],
    queryFn: () =>
      apiGet<TradeTapeResponse>(`/api/trade/tape/${symbol}`, undefined, { min_usd: 25_000, limit: 40 }),
    refetchInterval: 5_000,
    staleTime: 3_000,
    enabled: showTape,
  });

  const fundingQ = useQuery({
    queryKey: ['trade-funding', symbol],
    queryFn: () => apiGet<TradeFunding>(`/api/trade/funding/${symbol}`),
    refetchInterval: 60_000,
    staleTime: 30_000,
    // Funding is a perp-only concept; tolerate 503 quietly without retry storm.
    retry: 0,
  });

  const aiReadM = useMutation({
    mutationFn: () =>
      apiPost<TradeAiRead>(`/api/trade/ai_read/${symbol}`, {
        tf,
        toggles: { order_blocks: false, liq_heatmap: showWalls, big_prints: showTape },
        // Server fills tape stats from its own buffer when omitted.
      }),
  });

  // Pull-to-refresh state is intentionally driven by a manual flag — NOT by
  // any react-query `isFetching`. With the ticker polling every 1 second the
  // RefreshControl prop would flicker true→false constantly, and iOS reads
  // that flicker as "spinner is showing" which yanks the ScrollView back to
  // the top mid-scroll. We flip the flag for ~600ms on user pull and let it
  // settle independently of background polling.
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const onRefresh = useCallback(() => {
    setIsRefreshing(true);
    candlesQ.refetch();
    tickerQ.refetch();
    if (showFvg)   fvgQ.refetch();
    if (showTape)  tapeQ.refetch();
    wallsQ.refetch();
    fundingQ.refetch();
    setTimeout(() => setIsRefreshing(false), 600);
  }, [candlesQ, tickerQ, fvgQ, tapeQ, wallsQ, fundingQ, showFvg, showTape]);

  // ─── Chart data ──────────────────────────────────────────────────────────
  const chartCandles = candlesQ.data?.candles || [];
  const { width: screenW } = useWindowDimensions();
  const chartW = Math.max(screenW - spacing.lg * 2 - 2, 200);

  // Merge the live ticker price into the trailing (forming) candle so the
  // chart visibly ticks every second between full candle refreshes. We only
  // mutate the last candle's close + extend its high/low — never insert new
  // candles (that's the timeframe's job).
  const liveCandles = useMemo(() => {
    const px = tickerQ.data?.price;
    if (!chartCandles.length || px == null || !Number.isFinite(px)) return chartCandles;
    const out = chartCandles.slice();
    const last = { ...out[out.length - 1] };
    last.close = px;
    last.high  = Math.max(last.high, px);
    last.low   = Math.min(last.low,  px);
    out[out.length - 1] = last;
    return out;
  }, [chartCandles, tickerQ.data?.price]);

  const chartZones = useMemo<ChartZone[]>(() => {
    if (!showFvg) return [];
    const gaps = fvgQ.data?.gaps || [];
    if (chartCandles.length === 0 || gaps.length === 0) return [];
    const minTime = chartCandles[0].time;
    return gaps
      .filter((g) => g.time >= minTime)
      .map((g) => ({
        fromTime: g.time,
        toTime:   g.filled_at || undefined,
        top:      g.top,
        bottom:   g.bottom,
        side:     g.side,
        dim:      g.filled,
      }));
  }, [showFvg, fvgQ.data, chartCandles]);

  // Wall lines: draw the top 3 bids + top 3 asks as horizontal dashed lines
  // on the chart so the user can see support/resistance walls in price-context
  // (matching the web /trade page's "Liq" overlay).
  const chartPriceLines = useMemo<ChartPriceLine[]>(() => {
    if (!showWalls || !wallsQ.data) return [];
    const buys  = (wallsQ.data.top_buys || []).slice(0, 3);
    const sells = (wallsQ.data.top_sells || []).slice(0, 3);
    const toLine = (w: TradeWall): ChartPriceLine => ({
      price: w.price,
      side:  w.side,
      label: fmtUsd(w.size_usd),
    });
    return [...buys.map(toLine), ...sells.map(toLine)];
  }, [showWalls, wallsQ.data]);

  const fvgCounts = useMemo(() => {
    const gaps = fvgQ.data?.gaps || [];
    const unfilled = gaps.filter((g) => !g.filled);
    return {
      bull: unfilled.filter((g) => g.side === 'bull').length,
      bear: unfilled.filter((g) => g.side === 'bear').length,
      total: unfilled.length,
    };
  }, [fvgQ.data]);

  // Tape flow summary (used in chip + AI request hint)
  const tapeFlow = useMemo(() => {
    const tr = tapeQ.data?.trades || [];
    const buys  = tr.filter((t) => t.side === 'buy');
    const sells = tr.filter((t) => t.side === 'sell');
    const buyUsd  = buys.reduce((s, t) => s + t.usd, 0);
    const sellUsd = sells.reduce((s, t) => s + t.usd, 0);
    return { count: tr.length, buyUsd, sellUsd, buys: buys.length, sells: sells.length };
  }, [tapeQ.data]);

  const aiPlan = useMemo<AiPlan | null>(() => {
    if (!aiReadM.data?.summary) return null;
    return parseAiPlan(aiReadM.data.summary);
  }, [aiReadM.data]);

  const ticker = tickerQ.data;
  const walls = wallsQ.data;
  const livePrice = ticker?.price || (chartCandles.length ? chartCandles[chartCandles.length - 1].close : 0);

  const onPickSymbol = useCallback((next: string) => {
    if (next === symbol) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setSymbol(next);
    aiReadM.reset();
  }, [symbol, aiReadM]);

  const onPickTf = useCallback((next: string) => {
    if (next === tf) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setTf(next);
    aiReadM.reset();
  }, [tf, aiReadM]);

  const onToggleFvg = useCallback(() => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setShowFvg((v) => !v);
  }, []);
  const onToggleWalls = useCallback(() => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setShowWalls((v) => !v);
  }, []);
  const onToggleTape = useCallback(() => {
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    setShowTape((v) => !v);
  }, []);

  const onRunAiRead = useCallback(() => {
    if (Platform.OS !== 'web') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    aiReadM.mutate();
  }, [aiReadM]);

  return (
    <Screen
      title="Trade"
      subtitle="Live charts, FVG zones, walls, tape & AI read."
      ambient="cyan"
      refreshing={isRefreshing}
      onRefresh={onRefresh}
    >
      {/* ─── Coin picker ──────────────────────────────────────────────── */}
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

      {/* ─── Timeframe + overlay toggles ───────────────────────────────── */}
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
      </View>

      {/* Overlay toggle row — FVG · Walls · Tape (mirror the web /trade strip) */}
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.toggleRow}
      >
        <OverlayToggle
          label="FVG"
          icon="layers"
          active={showFvg}
          countLabel={
            showFvg && fvgCounts.total > 0
              ? `${fvgCounts.bull}↑/${fvgCounts.bear}↓`
              : undefined
          }
          onPress={onToggleFvg}
        />
        <OverlayToggle
          label="Walls"
          icon="bar-chart"
          active={showWalls}
          countLabel={
            showWalls && walls
              ? `${(walls.top_buys?.length || 0) + (walls.top_sells?.length || 0)}`
              : undefined
          }
          onPress={onToggleWalls}
        />
        <OverlayToggle
          label="Tape"
          icon="pulse"
          active={showTape}
          countLabel={
            showTape && tapeFlow.count > 0 ? `${tapeFlow.count}` : undefined
          }
          onPress={onToggleTape}
        />
      </ScrollView>

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
            candles={liveCandles}
            zones={chartZones}
            priceLines={chartPriceLines}
            width={chartW}
            height={320}
            symbol={`${symbol}USDT`}
            tf={tf}
            showOhlcLegend
            livePrice={tickerQ.data?.price}
          />
        )}
      </View>

      {/* ─── Pressure + best zone ───────────────────────────────────── */}
      {walls ? (
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
      ) : null}

      {walls?.best_zone_to_watch ? (
        <View style={styles.zoneCard}>
          <Ionicons name="flag" size={14} color={colors.accent} />
          <Text style={styles.zoneText}>{walls.best_zone_to_watch}</Text>
        </View>
      ) : null}

      {/* ─── Market context strip (vol + liquidations + funding) ─────── */}
      {(walls || fundingQ.data) ? (
        <View style={{ marginTop: spacing.lg }}>
          <SectionLabel label="Market context" />
          <View style={styles.contextGrid}>
            {walls ? (
              <>
                <ContextCell
                  label="24H VOLUME"
                  value={walls.volume_24h_usd ? fmtUsd(walls.volume_24h_usd) : '—'}
                />
                <ContextCell
                  label="1H VOLUME"
                  value={walls.volume_1h_usd ? fmtUsd(walls.volume_1h_usd) : '—'}
                />
                <ContextCell
                  label="LONG LIQ 24H"
                  value={walls.liq_24h_long_usd ? fmtUsd(walls.liq_24h_long_usd) : '—'}
                  tone={colors.negative}
                />
                <ContextCell
                  label="SHORT LIQ 24H"
                  value={walls.liq_24h_short_usd ? fmtUsd(walls.liq_24h_short_usd) : '—'}
                  tone={colors.positive}
                />
              </>
            ) : null}
            {fundingQ.data?.funding_rate_pct != null ? (
              <ContextCell
                label="FUNDING (8H)"
                value={`${fundingQ.data.funding_rate_pct >= 0 ? '+' : ''}${fundingQ.data.funding_rate_pct.toFixed(4)}%`}
                tone={fundingQ.data.funding_rate_pct >= 0 ? colors.positive : colors.negative}
                hint={fundingQ.data.funding_exchange}
              />
            ) : null}
            {fundingQ.data?.open_interest_usd != null ? (
              <ContextCell
                label="OPEN INTEREST"
                value={fmtUsd(fundingQ.data.open_interest_usd)}
                hint={
                  fundingQ.data.oi_change_24h_pct != null
                    ? `${fundingQ.data.oi_change_24h_pct >= 0 ? '+' : ''}${fundingQ.data.oi_change_24h_pct.toFixed(2)}% 24h`
                    : undefined
                }
              />
            ) : null}
          </View>
        </View>
      ) : null}

      {/* ─── AI Trade Read ──────────────────────────────────────────── */}
      <View style={{ marginTop: spacing.lg }}>
        <SectionLabel label="AI trade read" />
      </View>
      <View style={styles.aiContainer}>
        {!aiReadM.data && !aiReadM.isPending && !aiReadM.isError ? (
          <View style={styles.aiIntroCard}>
            <View style={styles.aiHeader}>
              <Ionicons name="sparkles" size={14} color={colors.violet} />
              <Text style={styles.aiHeaderText}>CLAUDE-POWERED CHART READ</Text>
            </View>
            <Text style={styles.aiBody}>
              Generates a structured trade plan (bias · entry · stop · TP1/TP2 · odds · leverage)
              from the live chart, FVGs, walls and big-prints flow.
            </Text>
            <Pressable
              onPress={onRunAiRead}
              style={({ pressed }) => [styles.aiBtn, pressed && { opacity: 0.85 }]}
            >
              <Ionicons name="flash" size={14} color={colors.accentText} />
              <Text style={styles.aiBtnText}>Generate AI read</Text>
            </Pressable>
          </View>
        ) : null}

        {aiReadM.isPending ? (
          <View style={styles.aiLoadingCard}>
            <ActivityIndicator size="small" color={colors.violet} />
            <Text style={styles.loadingText}>Reading the chart…</Text>
          </View>
        ) : null}

        {aiReadM.isError ? (
          <View style={styles.errorCard}>
            <Ionicons name="warning" size={18} color={colors.warning} />
            <View style={{ flex: 1 }}>
              <Text style={styles.errorText}>AI read failed.</Text>
              <Pressable onPress={onRunAiRead}>
                <Text style={[styles.errorText, { color: colors.accent, marginTop: 4 }]}>Tap to retry</Text>
              </Pressable>
            </View>
          </View>
        ) : null}

        {aiPlan ? (
          <View style={styles.aiPlanCard}>
            <View style={styles.aiHeader}>
              <Ionicons name="sparkles" size={14} color={colors.violet} />
              <Text style={styles.aiHeaderText}>
                AI READ · {aiReadM.data?.tf?.toUpperCase()}
                {aiReadM.data?.cached ? '  · CACHED' : ''}
                {aiReadM.data?.fallback ? '  · FALLBACK' : ''}
              </Text>
              <Pressable onPress={onRunAiRead} style={styles.aiRefreshBtn}>
                <Ionicons name="refresh" size={13} color={colors.violet} />
              </Pressable>
            </View>

            {aiPlan.bias ? (
              <View style={styles.aiBiasRow}>
                <View style={[
                  styles.aiBiasChip,
                  { backgroundColor: aiPlan.bias === 'LONG' ? colors.positive : colors.negative },
                ]}>
                  <Text style={styles.aiBiasText}>{aiPlan.bias}</Text>
                </View>
                {aiPlan.entry ? (
                  <Text style={styles.aiEntry}>@ ${fmtPrice(parseFloat(aiPlan.entry))}</Text>
                ) : null}
                {aiPlan.orderType ? (
                  <Pill label={aiPlan.orderType} tone="neutral" small />
                ) : null}
              </View>
            ) : null}

            <View style={styles.aiPlanGrid}>
              {aiPlan.stop ? (
                <PlanCell label="STOP" value={`$${fmtPrice(parseFloat(aiPlan.stop))}`} tone={colors.negative} />
              ) : null}
              {aiPlan.tp1 ? (
                <PlanCell label="TP1" value={`$${fmtPrice(parseFloat(aiPlan.tp1))}`} tone={colors.positive} />
              ) : null}
              {aiPlan.tp2 ? (
                <PlanCell label="TP2" value={`$${fmtPrice(parseFloat(aiPlan.tp2))}`} tone={colors.positive} />
              ) : null}
              {aiPlan.leverage ? (
                <PlanCell label="LEVERAGE" value={aiPlan.leverage} />
              ) : null}
            </View>

            {aiPlan.oddsUp != null && aiPlan.oddsDown != null ? (
              <View style={styles.oddsBar}>
                <View style={[styles.oddsFill, { flex: aiPlan.oddsUp,   backgroundColor: colors.positive }]}>
                  <Text style={styles.oddsLabel}>UP {aiPlan.oddsUp}%</Text>
                </View>
                <View style={[styles.oddsFill, { flex: aiPlan.oddsDown, backgroundColor: colors.negative }]}>
                  <Text style={styles.oddsLabel}>DOWN {aiPlan.oddsDown}%</Text>
                </View>
              </View>
            ) : null}

            {aiPlan.rationale ? (
              <Text style={styles.aiRationale}>{aiPlan.rationale}</Text>
            ) : null}

            {aiReadM.data?.sources_used && aiReadM.data.sources_used.length ? (
              <View style={styles.aiSources}>
                {aiReadM.data.sources_used.map((s) => (
                  <Pill key={s} label={s.toUpperCase()} tone="violet" small />
                ))}
              </View>
            ) : null}
          </View>
        ) : null}
      </View>

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
          <View style={styles.wallsBlock}>
            <Text style={styles.wallsBlockTitle}>
              <Text style={{ color: colors.positive }}>BIDS</Text>
              <Text style={{ color: colors.textMute }}>  · support below price</Text>
            </Text>
            {(walls.top_buys || []).slice(0, 5).map((w, i) => (
              <WallRow key={`buy-${i}`} wall={w} />
            ))}
            {(!walls.top_buys || walls.top_buys.length === 0) ? (
              <Text style={styles.wallsEmpty}>No major bid walls detected.</Text>
            ) : null}
          </View>

          <View style={styles.wallsBlock}>
            <Text style={styles.wallsBlockTitle}>
              <Text style={{ color: colors.negative }}>ASKS</Text>
              <Text style={{ color: colors.textMute }}>  · resistance above price</Text>
            </Text>
            {(walls.top_sells || []).slice(0, 5).map((w, i) => (
              <WallRow key={`sell-${i}`} wall={w} />
            ))}
            {(!walls.top_sells || walls.top_sells.length === 0) ? (
              <Text style={styles.wallsEmpty}>No major ask walls detected.</Text>
            ) : null}
          </View>

          {walls.ai_summary ? (
            <View style={styles.aiCard}>
              <View style={styles.aiHeader}>
                <Ionicons name="sparkles" size={14} color={colors.violet} />
                <Text style={styles.aiHeaderText}>WALLS READ</Text>
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

      {/* ─── Big Prints / Live Tape ─────────────────────────────────── */}
      {showTape ? (
        <>
          <View style={{ marginTop: spacing.xl }}>
            <SectionLabel label={`Big prints · ≥ $25K`} />
          </View>

          {tapeFlow.count > 0 ? (
            <View style={styles.tapeFlowCard}>
              <View style={styles.tapeFlowItem}>
                <Text style={styles.tapeFlowLabel}>BUYS</Text>
                <Text style={[styles.tapeFlowValue, { color: colors.positive }]}>
                  {fmtUsd(tapeFlow.buyUsd)}
                </Text>
                <Text style={styles.tapeFlowCount}>{tapeFlow.buys} prints</Text>
              </View>
              <View style={styles.tapeFlowDivider} />
              <View style={styles.tapeFlowItem}>
                <Text style={styles.tapeFlowLabel}>SELLS</Text>
                <Text style={[styles.tapeFlowValue, { color: colors.negative }]}>
                  {fmtUsd(tapeFlow.sellUsd)}
                </Text>
                <Text style={styles.tapeFlowCount}>{tapeFlow.sells} prints</Text>
              </View>
            </View>
          ) : null}

          {tapeQ.isLoading ? (
            <View style={styles.loadingCard}>
              <ActivityIndicator size="small" color={colors.accent} />
              <Text style={styles.loadingText}>Loading tape…</Text>
            </View>
          ) : (tapeQ.data?.trades || []).length === 0 ? (
            <View style={styles.tapeEmpty}>
              <Text style={styles.tapeEmptyText}>
                No big prints yet. Tape is live — large trades will stream in here.
              </Text>
            </View>
          ) : (
            <View style={styles.tapeList}>
              {(tapeQ.data?.trades || []).slice(0, 15).map((t) => (
                <TapeRow key={t.id} trade={t} />
              ))}
            </View>
          )}
        </>
      ) : null}

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

function OverlayToggle({
  label, icon, active, countLabel, onPress,
}: {
  label: string;
  icon: keyof typeof Ionicons.glyphMap;
  active: boolean;
  countLabel?: string;
  onPress: () => void;
}) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.overlayToggle,
        active && styles.overlayToggleActive,
        pressed && { opacity: 0.85 },
      ]}
    >
      <Ionicons
        name={icon}
        size={13}
        color={active ? colors.accent : colors.textMute}
      />
      <Text style={[styles.overlayToggleText, active && { color: colors.accent }]}>
        {label}
      </Text>
      {countLabel ? (
        <View style={styles.overlayCountChip}>
          <Text style={styles.overlayCountText}>{countLabel}</Text>
        </View>
      ) : null}
    </Pressable>
  );
}

function ContextCell({
  label, value, hint, tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: string;
}) {
  return (
    <View style={styles.contextCell}>
      <Text style={styles.contextLabel}>{label}</Text>
      <Text style={[styles.contextValue, tone ? { color: tone } : null]}>{value}</Text>
      {hint ? <Text style={styles.contextHint}>{hint}</Text> : null}
    </View>
  );
}

function PlanCell({
  label, value, tone,
}: {
  label: string;
  value: string;
  tone?: string;
}) {
  return (
    <View style={styles.planCell}>
      <Text style={styles.planLabel}>{label}</Text>
      <Text style={[styles.planValue, tone ? { color: tone } : null]}>{value}</Text>
    </View>
  );
}

function WallRow({ wall }: { wall: TradeWall }) {
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

function TapeRow({ trade }: { trade: TradeTapeRow }) {
  const isBuy = trade.side === 'buy';
  const tone = isBuy ? colors.positive : colors.negative;
  return (
    <View style={styles.tapeRow}>
      <View style={[styles.tapeSideBar, { backgroundColor: tone }]} />
      <View style={styles.tapeMain}>
        <Text style={[styles.tapeSide, { color: tone }]}>
          {isBuy ? 'BUY' : 'SELL'}
        </Text>
        <Text style={styles.tapePrice}>${fmtPrice(trade.price)}</Text>
      </View>
      <View style={styles.tapeRight}>
        <Text style={styles.tapeSize}>{fmtUsd(trade.usd)}</Text>
        <Text style={styles.tapeAge}>{fmtRelMs(trade.ts)}</Text>
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
  tickerTopRow: { flexDirection: 'row', alignItems: 'flex-end' },
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

  // TF + overlay row
  tfRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.md,
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
  toggleRow: {
    gap: 8,
    paddingVertical: spacing.sm,
  },
  overlayToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
  },
  overlayToggleActive: {
    borderColor: 'rgba(34,211,238,0.32)',
    backgroundColor: colors.accentDim,
  },
  overlayToggleText: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.5,
  },
  overlayCountChip: {
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderRadius: radius.pill,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  overlayCountText: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 10.5,
    fontVariant: ['tabular-nums'],
  },

  // Loading / error
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
    marginTop: spacing.md,
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

  // Context grid
  contextGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  contextCell: {
    flexBasis: '48%',
    flexGrow: 1,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    padding: spacing.sm,
  },
  contextLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
  },
  contextValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
    marginTop: 3,
  },
  contextHint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 10,
    marginTop: 2,
  },

  // AI Trade Read
  aiContainer: {},
  aiIntroCard: {
    backgroundColor: colors.violetDim,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
    padding: spacing.md,
  },
  aiLoadingCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: colors.violetDim,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
    padding: spacing.lg,
    justifyContent: 'center',
  },
  aiPlanCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.32)',
    padding: spacing.md,
    ...glow.card,
  },
  aiHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  aiHeaderText: {
    color: colors.violet,
    fontFamily: font.black,
    fontSize: 10.5,
    letterSpacing: 0.7,
    flex: 1,
  },
  aiRefreshBtn: {
    padding: 4,
    borderRadius: radius.pill,
  },
  aiBody: {
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: spacing.md,
  },
  aiBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    backgroundColor: colors.accent,
    paddingVertical: 11,
    borderRadius: radius.pill,
  },
  aiBtnText: {
    color: colors.accentText,
    fontFamily: font.bold,
    fontSize: 13,
    letterSpacing: 0.4,
  },
  aiBiasRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: spacing.md,
  },
  aiBiasChip: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.pill,
  },
  aiBiasText: {
    color: '#04111a',
    fontFamily: font.black,
    fontSize: 12,
    letterSpacing: 0.6,
  },
  aiEntry: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 15,
    fontVariant: ['tabular-nums'],
  },
  aiPlanGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: spacing.sm,
  },
  planCell: {
    flexBasis: '48%',
    flexGrow: 1,
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderRadius: radius.md,
    padding: spacing.sm,
  },
  planLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
  },
  planValue: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
    marginTop: 3,
  },
  oddsBar: {
    flexDirection: 'row',
    height: 22,
    borderRadius: radius.pill,
    overflow: 'hidden',
    marginVertical: spacing.sm,
  },
  oddsFill: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  oddsLabel: {
    color: '#04111a',
    fontFamily: font.black,
    fontSize: 10.5,
    letterSpacing: 0.5,
  },
  aiRationale: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 12.5,
    lineHeight: 17.5,
    marginTop: spacing.sm,
  },
  aiSources: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 5,
    marginTop: spacing.md,
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

  aiCard: {
    marginTop: spacing.md,
    backgroundColor: colors.violetDim,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: 'rgba(167,139,250,0.28)',
    padding: spacing.md,
  },
  exchPills: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 5,
    marginTop: spacing.md,
    justifyContent: 'center',
  },

  // Tape
  tapeFlowCard: {
    flexDirection: 'row',
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    marginBottom: spacing.sm,
  },
  tapeFlowItem: { flex: 1, alignItems: 'center' },
  tapeFlowLabel: {
    color: '#8b95b3',
    fontFamily: font.bold,
    fontSize: 9.5,
    letterSpacing: 0.7,
  },
  tapeFlowValue: {
    fontFamily: font.black,
    fontSize: 17,
    fontVariant: ['tabular-nums'],
    marginTop: 4,
  },
  tapeFlowCount: {
    color: colors.textMute,
    fontFamily: font.semibold,
    fontSize: 10.5,
    marginTop: 2,
  },
  tapeFlowDivider: {
    width: 1,
    backgroundColor: colors.border,
    marginHorizontal: spacing.md,
  },
  tapeEmpty: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    borderStyle: 'dashed',
    padding: spacing.lg,
  },
  tapeEmptyText: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12.5,
    textAlign: 'center',
    lineHeight: 18,
  },
  tapeList: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  tapeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.md,
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: 'rgba(255,255,255,0.04)',
    gap: 10,
  },
  tapeSideBar: {
    width: 3,
    height: 22,
    borderRadius: 2,
  },
  tapeMain: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 8,
  },
  tapeSide: {
    fontFamily: font.black,
    fontSize: 10.5,
    letterSpacing: 0.6,
    width: 32,
  },
  tapePrice: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 12.5,
    fontVariant: ['tabular-nums'],
  },
  tapeRight: { alignItems: 'flex-end' },
  tapeSize: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 12.5,
    fontVariant: ['tabular-nums'],
  },
  tapeAge: {
    color: colors.textMute,
    fontFamily: font.semibold,
    fontSize: 9.5,
    marginTop: 1,
    fontVariant: ['tabular-nums'],
  },
});
