import React, { useCallback, useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Pressable,
  Linking,
  useWindowDimensions,
} from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';

import { Pill } from '@/components/Pill';
import { StatCard } from '@/components/StatCard';
import { EquityCurve } from '@/components/EquityCurve';
import { PrimaryButton } from '@/components/PrimaryButton';
import { EmptyState } from '@/components/EmptyState';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import {
  apiGet,
  apiPostFlex,
  type Strategy,
  type BacktestResult,
} from '@/lib/api';

// ─── Helpers ───────────────────────────────────────────────────────────────

function fmtPnl(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toFixed(digits);
}

function pnlTone(v: number | null | undefined): 'positive' | 'negative' | 'neutral' {
  if (v === null || v === undefined) return 'neutral';
  if (v > 0.01) return 'positive';
  if (v < -0.01) return 'negative';
  return 'neutral';
}

// ─── Period toggle ─────────────────────────────────────────────────────────

function PeriodToggle({
  value,
  onChange,
  disabled,
}: {
  value: 30 | 90;
  onChange: (v: 30 | 90) => void;
  disabled: boolean;
}) {
  return (
    <View style={styles.toggle}>
      {[30, 90].map((d) => {
        const active = value === d;
        return (
          <Pressable
            key={d}
            disabled={disabled}
            onPress={() => {
              Haptics.selectionAsync().catch(() => {});
              onChange(d as 30 | 90);
            }}
            style={({ pressed }) => [
              styles.toggleSeg,
              active && styles.toggleSegActive,
              pressed && !disabled && { opacity: 0.85 },
              disabled && { opacity: 0.5 },
            ]}
          >
            <Text style={[styles.toggleSegText, active && styles.toggleSegTextActive]}>
              {d} days
            </Text>
            <Text style={[styles.toggleSegSub, active && { color: colors.accentText }]}>
              {d === 30 ? '~1h candles' : '~1h candles'}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

// ─── Loading panel ─────────────────────────────────────────────────────────

function RunningPanel({ days }: { days: 30 | 90 }) {
  return (
    <View style={[styles.runningCard, glow.card]}>
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <SvgLinearGradient id="bt-running" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor={colors.card} />
            <Stop offset="0.6" stopColor={colors.card} />
            <Stop offset="1" stopColor={colors.card} />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill="url(#bt-running)" />
      </Svg>
      <View style={styles.runningInner}>
        <ActivityIndicator color={colors.accent} size="large" />
        <Text style={styles.runningTitle}>Running backtest…</Text>
        <Text style={styles.runningSub}>
          Replaying {days} days of historical candles through your strategy.
          {'\n'}
          This usually takes 10–30 seconds — please keep the screen open.
        </Text>
        <View style={styles.runningSteps}>
          <View style={styles.runningStep}>
            <Ionicons name="cloud-download-outline" size={13} color={colors.accent} />
            <Text style={styles.runningStepText}>Fetching klines</Text>
          </View>
          <View style={styles.runningStep}>
            <Ionicons name="analytics-outline" size={13} color={colors.accent} />
            <Text style={styles.runningStepText}>Replaying signals</Text>
          </View>
          <View style={styles.runningStep}>
            <Ionicons name="trending-up" size={13} color={colors.accent} />
            <Text style={styles.runningStepText}>Computing P&L</Text>
          </View>
        </View>
      </View>
    </View>
  );
}

// ─── Pro paywall ───────────────────────────────────────────────────────────

function ProPaywall({ message, onUpgrade }: { message?: string; onUpgrade: () => void }) {
  return (
    <View style={[styles.paywall, glow.card]}>
      <Svg style={StyleSheet.absoluteFill}>
        <Defs>
          <SvgLinearGradient id="bt-paywall" x1="0" y1="0" x2="1" y2="1">
            <Stop offset="0" stopColor="#1A1C20" />
            <Stop offset="0.5" stopColor="#1A1C20" />
            <Stop offset="1" stopColor="#1A1C20" />
          </SvgLinearGradient>
        </Defs>
        <Rect width="100%" height="100%" fill="url(#bt-paywall)" />
      </Svg>
      <View style={styles.paywallInner}>
        <View style={styles.paywallBadge}>
          <Ionicons name="diamond" size={14} color={colors.violet} />
          <Text style={styles.paywallBadgeText}>PRO FEATURE</Text>
        </View>
        <Text style={styles.paywallTitle}>Backtests are a Pro feature</Text>
        <Text style={styles.paywallBody}>
          {message ||
            'Replay your strategy against 30 or 90 days of real historical data, with realistic fee, slippage and funding modelling.'}
        </Text>
        <View style={styles.paywallBullets}>
          {[
            'Realistic fees (0.04%) + slippage modelling',
            'Liquidation simulation at your leverage',
            'Win rate, profit factor, equity curve',
            'AI-powered improvement suggestions',
          ].map((b) => (
            <View key={b} style={styles.paywallBulletRow}>
              <Ionicons name="checkmark-circle" size={14} color={colors.violet} />
              <Text style={styles.paywallBulletText}>{b}</Text>
            </View>
          ))}
        </View>
        <PrimaryButton
          label="Upgrade to Pro"
          variant="primary"
          icon={<Ionicons name="diamond" size={15} color={colors.accentText} />}
          onPress={onUpgrade}
        />
      </View>
    </View>
  );
}

// ─── Results ───────────────────────────────────────────────────────────────

function ResultsPanel({
  result,
  chartW,
}: {
  result: BacktestResult;
  chartW: number;
}) {
  const stats = result.stats || {};
  const equity = result.equity_curve || [];
  const equityValues = useMemo(() => equity.map((p) => p.y).filter((v) => Number.isFinite(v)), [equity]);
  const totalPnl = stats.total_pnl ?? 0;
  const wr = stats.win_rate ?? 0;
  const closed = stats.closed_trades ?? stats.total_trades ?? 0;
  const pf = stats.profit_factor ?? 0;
  const dd = stats.max_drawdown ?? 0;
  const trades = result.trades || [];
  const wins = trades.filter((t) => t.outcome === 'WIN').length;
  const losses = trades.filter((t) => t.outcome === 'LOSS').length;

  return (
    <View>
      {/* Hero result card */}
      <View style={[styles.resultHero, glow.accent]}>
        <Svg style={StyleSheet.absoluteFill}>
          <Defs>
            <SvgLinearGradient id="bt-result" x1="0" y1="0" x2="1" y2="1">
              <Stop offset="0" stopColor={colors.card} />
              <Stop offset="0.55" stopColor={colors.card} />
              <Stop offset="1" stopColor={colors.card} />
            </SvgLinearGradient>
          </Defs>
          <Rect width="100%" height="100%" fill="url(#bt-result)" />
        </Svg>
        <View style={styles.resultHeroInner}>
          <Text style={styles.resultLabel}>BACKTEST P&L · {result.days ?? 30}d</Text>
          <Text
            style={[
              styles.resultBig,
              {
                color:
                  totalPnl > 0.01
                    ? colors.positive
                    : totalPnl < -0.01
                      ? colors.negative
                      : colors.text,
              },
            ]}
          >
            {fmtPnl(totalPnl)}
          </Text>
          <View style={styles.resultMeta}>
            <View style={styles.resultMetaItem}>
              <Text style={styles.resultMetaLabel}>WIN RATE</Text>
              <Text style={styles.resultMetaVal}>{closed > 0 ? `${wr.toFixed(1)}%` : '—'}</Text>
            </View>
            <View style={styles.resultMetaSep} />
            <View style={styles.resultMetaItem}>
              <Text style={styles.resultMetaLabel}>TRADES</Text>
              <Text style={styles.resultMetaVal}>{closed}</Text>
            </View>
            <View style={styles.resultMetaSep} />
            <View style={styles.resultMetaItem}>
              <Text style={styles.resultMetaLabel}>PROFIT FACTOR</Text>
              <Text style={styles.resultMetaVal}>{pf > 0 ? fmtNum(pf, 2) : '—'}</Text>
            </View>
          </View>
        </View>
      </View>

      {/* Equity curve */}
      {equityValues.length > 1 ? (
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.section}>Equity curve</Text>
          <EquityCurve values={equityValues} width={chartW} height={170} />
        </View>
      ) : null}

      {/* Stat grid */}
      <View style={[styles.statRow, { marginTop: spacing.lg }]}>
        <StatCard
          label="Wins"
          value={`${wins}`}
          tone="positive"
          sub={`avg ${fmtPnl(stats.avg_win_pct ?? 0)}`}
          compact
        />
        <View style={{ width: spacing.md }} />
        <StatCard
          label="Losses"
          value={`${losses}`}
          tone="negative"
          sub={`avg ${fmtPnl(stats.avg_loss_pct ?? 0)}`}
          compact
        />
        <View style={{ width: spacing.md }} />
        <StatCard
          label="Max DD"
          value={fmtPnl(dd)}
          tone="warning"
          compact
        />
      </View>

      {/* Sample trades */}
      {trades.length > 0 ? (
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.section}>Sample trades · last {Math.min(trades.length, 12)}</Text>
          <View style={styles.tradeList}>
            {trades.slice(-12).reverse().map((t, i) => {
              const tone = pnlTone(t.pnl_pct);
              const c =
                tone === 'positive' ? colors.positive : tone === 'negative' ? colors.negative : colors.text;
              return (
                <View
                  key={`bt-t-${i}`}
                  style={[styles.tradeRow, i < Math.min(trades.length, 12) - 1 && styles.tradeRowDiv]}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={styles.tradeSymbol}>{t.symbol || result.symbol || '—'}</Text>
                    <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6, marginTop: 3 }}>
                      <Pill
                        label={t.direction || 'LONG'}
                        tone={t.direction === 'SHORT' ? 'negative' : 'positive'}
                        small
                      />
                      <Text style={styles.tradeDate}>{t.entry_date || '—'}</Text>
                    </View>
                  </View>
                  <View style={{ alignItems: 'flex-end' }}>
                    <Text style={[styles.tradePnl, { color: c }]}>{fmtPnl(t.pnl_pct ?? 0)}</Text>
                    <Text style={styles.tradeOutcome}>{t.outcome || '—'}</Text>
                  </View>
                </View>
              );
            })}
          </View>
        </View>
      ) : null}

      {/* Footnotes */}
      <View style={[styles.footnote, { marginTop: spacing.lg }]}>
        <Ionicons name="information-circle-outline" size={14} color={colors.textMute} />
        <Text style={styles.footnoteText}>
          Past performance is not indicative of future results. Backtests assume realistic fees
          (0.04%) and slippage but cannot capture exchange-side liquidity gaps.
        </Text>
      </View>
    </View>
  );
}

// ─── Screen ────────────────────────────────────────────────────────────────

export default function BacktestScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const sid = Number(id);
  const router = useRouter();
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const { width } = useWindowDimensions();
  const chartW = width - spacing.lg * 2 - spacing.sm * 2;

  const [days, setDays] = useState<30 | 90>(30);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Strategy lookup — piggyback on the cached list (already loaded by other tabs).
  const listQ = useQuery({
    queryKey: ['strategies', uid],
    queryFn: () => apiGet<Strategy[]>('/api/strategies', uid),
    enabled: !!uid,
  });
  const strategy = useMemo(
    () => (listQ.data || []).find((s) => s.id === sid),
    [listQ.data, sid],
  );

  const runBacktest = useCallback(async () => {
    if (!uid || !strategy) return;
    // Guard: backend requires a config blob with at least entry conditions —
    // sending {} would 500 with a confusing "no entry conditions" error. Block
    // the run client-side and surface a clear "edit on web" message instead.
    const cfg = strategy.config;
    const hasConfig =
      !!cfg &&
      typeof cfg === 'object' &&
      Object.keys(cfg as Record<string, unknown>).length > 0;
    if (!hasConfig) {
      setErrorMsg(
        "This strategy doesn't have a backtest-ready config. Open it on the web portal to finish setup, then try again.",
      );
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
      return;
    }
    setRunning(true);
    setResult(null);
    setErrorMsg(null);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});

    try {
      const resp = await apiPostFlex<BacktestResult>(
        '/api/backtest/run',
        { uid, config: cfg, days },
        uid,
      );

      // 408 — backtest exceeded the engine's 90s budget
      if (resp.status === 408 || resp.body?.error === 'TIMEOUT') {
        setErrorMsg(
          resp.body?.message ||
            'Backtest timed out. Try a shorter window (30 days) or a strategy with fewer indicators.',
        );
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning).catch(() => {});
        return;
      }
      // Other server-side error path
      if (!resp.ok || resp.body?.error) {
        setErrorMsg(
          resp.body?.message ||
            `Backtest failed (HTTP ${resp.status}). Please try again in a moment.`,
        );
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
        return;
      }

      // Success
      setResult(resp.body);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    } catch (e) {
      setErrorMsg(
        (e as Error)?.message ||
          "Couldn't reach the backtest engine. Check your connection and try again.",
      );
    } finally {
      setRunning(false);
    }
  }, [uid, strategy, days]);

  // ── Loading list ──
  if (listQ.isLoading || (!listQ.data && listQ.isFetching)) {
    return (
      <>
        <Stack.Screen options={{ title: 'Backtest' }} />
        <View style={[styles.center, { backgroundColor: colors.bg, flex: 1 }]}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      </>
    );
  }

  if (!strategy) {
    return (
      <>
        <Stack.Screen options={{ title: 'Backtest' }} />
        <EmptyState
          icon="alert-circle-outline"
          title="Strategy not found"
          hint="It may have been archived. Pick another strategy from the list."
        />
      </>
    );
  }

  return (
    <>
      <Stack.Screen options={{ title: 'Backtest' }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: colors.bg }}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 32 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Pressable
            onPress={() => router.back()}
            style={({ pressed }) => [styles.backBtn, pressed && { opacity: 0.6 }]}
            hitSlop={10}
          >
            <Ionicons name="chevron-back" size={20} color={colors.text} />
          </Pressable>
          <Text style={styles.title}>Backtest</Text>
          <Text style={styles.subtitle} numberOfLines={2}>
            {strategy.name}
          </Text>
        </View>

        {/* Period + Run */}
        <View style={{ marginTop: spacing.lg }}>
          <Text style={styles.section}>Replay window</Text>
          <PeriodToggle value={days} onChange={setDays} disabled={running} />
        </View>

        <View style={{ marginTop: spacing.lg }}>
          <PrimaryButton
            label={running ? 'Running…' : result ? 'Re-run backtest' : 'Run backtest'}
            onPress={runBacktest}
            loading={running}
            icon={!running ? <Ionicons name="play" size={15} color={colors.accentText} /> : undefined}
          />
        </View>

        {/* Body — running, error, or results */}
        <View style={{ marginTop: spacing.lg }}>
          {running ? (
            <RunningPanel days={days} />
          ) : errorMsg ? (
            <View style={styles.errorCard}>
              <Ionicons name="alert-circle" size={18} color={colors.warning} />
              <Text style={styles.errorText}>{errorMsg}</Text>
            </View>
          ) : result ? (
            <ResultsPanel result={result} chartW={chartW} />
          ) : (
            <View style={styles.emptyHint}>
              <Ionicons name="time-outline" size={20} color={colors.textDim} />
              <Text style={styles.emptyHintText}>
                Pick a window above and tap “Run backtest”. Results — equity curve, win rate, sample
                trades — will appear here.
              </Text>
            </View>
          )}
        </View>
      </ScrollView>

    </>
  );
}

// ─── Styles ────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  content: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xxl + 8,
    paddingBottom: spacing.xxl,
  },
  center: { alignItems: 'center', justifyContent: 'center', flex: 1, padding: spacing.xl },

  header: { paddingTop: spacing.sm },
  backBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: spacing.md,
  },
  title: { color: colors.text, fontFamily: font.black, fontSize: 30, letterSpacing: -0.8 },
  subtitle: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 14,
    marginTop: 4,
    lineHeight: 19,
  },

  section: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },

  // Period toggle
  toggle: {
    flexDirection: 'row',
    backgroundColor: colors.cardHi,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: 4,
    gap: 4,
  },
  toggleSeg: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 14,
    borderRadius: radius.md,
    alignItems: 'center',
  },
  toggleSegActive: {
    backgroundColor: colors.accent,
    ...glow.accent,
  },
  toggleSegText: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 14,
    letterSpacing: 0.3,
  },
  toggleSegTextActive: {
    color: colors.accentText,
  },
  toggleSegSub: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 10,
    marginTop: 2,
    letterSpacing: 0.3,
  },

  // Empty hint
  emptyHint: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    borderStyle: 'dashed',
    padding: spacing.lg,
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
  },
  emptyHintText: {
    flex: 1,
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 18,
  },

  // Running
  runningCard: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  runningInner: {
    padding: spacing.xl,
    alignItems: 'center',
  },
  runningTitle: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 17,
    marginTop: spacing.md,
    letterSpacing: -0.3,
  },
  runningSub: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
    marginTop: spacing.sm,
  },
  runningSteps: {
    flexDirection: 'row',
    gap: 12,
    marginTop: spacing.lg,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  runningStep: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    backgroundColor: 'rgba(255,255,255,0.10)',
    borderColor: 'rgba(255,255,255,0.10)',
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
  },
  runningStepText: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.3,
  },

  // Paywall
  paywall: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  paywallInner: { padding: spacing.xl },
  paywallBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    backgroundColor: colors.violetDim,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: radius.pill,
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
  },
  paywallBadgeText: {
    color: colors.violet,
    fontFamily: font.black,
    fontSize: 10,
    letterSpacing: 0.7,
  },
  paywallTitle: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 20,
    letterSpacing: -0.4,
    marginTop: spacing.md,
  },
  paywallBody: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    lineHeight: 20,
    marginTop: spacing.sm,
  },
  paywallBullets: {
    marginTop: spacing.lg,
    marginBottom: spacing.lg,
    gap: 8,
  },
  paywallBulletRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  paywallBulletText: {
    color: colors.text,
    fontFamily: font.medium,
    fontSize: 13,
    flex: 1,
  },

  // Error card
  errorCard: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'flex-start',
    backgroundColor: colors.warningDim,
    borderColor: 'rgba(251,191,36,0.32)',
    borderWidth: 1,
    borderRadius: radius.lg,
    padding: spacing.md,
  },
  errorText: {
    flex: 1,
    color: colors.text,
    fontFamily: font.medium,
    fontSize: 13,
    lineHeight: 18,
  },

  // Result hero
  resultHero: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.10)',
    overflow: 'hidden',
    backgroundColor: colors.card,
  },
  resultHeroInner: { padding: spacing.xl },
  resultLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 1.0,
  },
  resultBig: {
    fontFamily: font.black,
    fontSize: 48,
    letterSpacing: -1.6,
    marginTop: 8,
    fontVariant: ['tabular-nums'],
  },
  resultMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: 'rgba(0,0,0,0.32)',
    borderRadius: radius.lg,
    padding: spacing.md,
  },
  resultMetaItem: { flex: 1, alignItems: 'center' },
  resultMetaLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.7,
  },
  resultMetaVal: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 17,
    fontVariant: ['tabular-nums'],
    marginTop: 3,
  },
  resultMetaSep: {
    width: 1,
    backgroundColor: colors.divider,
    alignSelf: 'stretch',
    marginVertical: 4,
  },

  statRow: { flexDirection: 'row' },

  // Sample trades
  tradeList: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    overflow: 'hidden',
  },
  tradeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: 11,
  },
  tradeRowDiv: {
    borderBottomWidth: 1,
    borderBottomColor: colors.divider,
  },
  tradeSymbol: { color: colors.text, fontFamily: font.bold, fontSize: 14 },
  tradeDate: { color: colors.textMute, fontFamily: font.regular, fontSize: 11 },
  tradePnl: { fontFamily: font.bold, fontSize: 14, fontVariant: ['tabular-nums'] },
  tradeOutcome: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 9,
    letterSpacing: 0.5,
    marginTop: 2,
  },

  footnote: {
    flexDirection: 'row',
    gap: 8,
    alignItems: 'flex-start',
    paddingHorizontal: 4,
  },
  footnoteText: {
    flex: 1,
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11.5,
    lineHeight: 16,
  },
});
