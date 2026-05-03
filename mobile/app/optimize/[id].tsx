import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  ActivityIndicator,
  Pressable,
  Alert,
} from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

import { Pill } from '@/components/Pill';
import { PrimaryButton } from '@/components/PrimaryButton';
import { EmptyState } from '@/components/EmptyState';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiPostFlex } from '@/lib/api';

// ─── Types ────────────────────────────────────────────────────────────────

type BacktestStats = {
  total_signals?: number;
  closed_trades?: number;
  wins?: number;
  losses?: number;
  win_rate?: number;
  total_pnl?: number;
  total_pnl_simple?: number;
  max_drawdown?: number;
  profit_factor?: number | string;
  liquidations?: number;
};

type VariantPatch = {
  tp1?: number;
  sl?: number;
  leverage?: number;
  timeframe?: string;
};

type Variant = {
  label: string;
  tagline: string;
  patch: VariantPatch;
  stats: BacktestStats;
  score: number;
  delta_pnl: number;
  delta_dd: number;
  delta_win_rate: number;
  improved: boolean;
};

type OptimizeResult = {
  days: 30 | 90;
  baseline: { label: string; stats: BacktestStats; score: number };
  variants: Variant[];
  any_improved: boolean;
  ran_at: string;
};

type ApplyResult = { ok: boolean; applied: VariantPatch };

// ─── Helpers ──────────────────────────────────────────────────────────────

function fmtPnl(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function fmtDelta(v: number, suffix = '%'): string {
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}${suffix}`;
}

function pnlTone(v: number | null | undefined): 'positive' | 'negative' | 'neutral' {
  if (v === null || v === undefined) return 'neutral';
  if (v > 0.01) return 'positive';
  if (v < -0.01) return 'negative';
  return 'neutral';
}

// ─── Period toggle (30 / 90 days) ─────────────────────────────────────────

function PeriodToggle({
  value,
  onChange,
  disabled,
}: {
  value: 30 | 90;
  onChange: (v: 30 | 90) => void;
  disabled?: boolean;
}) {
  return (
    <View style={styles.toggleWrap}>
      {([30, 90] as const).map((d) => {
        const active = value === d;
        return (
          <Pressable
            key={d}
            disabled={disabled}
            onPress={() => {
              Haptics.selectionAsync();
              onChange(d);
            }}
            style={[styles.toggleBtn, active && styles.toggleBtnActive, disabled && { opacity: 0.5 }]}
          >
            <Text style={[styles.toggleTxt, active && styles.toggleTxtActive]}>
              Last {d} days
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

// ─── Patch summary chips ──────────────────────────────────────────────────

function PatchChips({ patch }: { patch: VariantPatch }) {
  const items: { label: string; value: string }[] = [];
  if (patch.tp1 !== undefined) items.push({ label: 'TP', value: `${patch.tp1}%` });
  if (patch.sl !== undefined) items.push({ label: 'SL', value: `${patch.sl}%` });
  if (patch.leverage !== undefined) items.push({ label: 'Lev', value: `${patch.leverage}×` });
  if (patch.timeframe !== undefined) items.push({ label: 'TF', value: patch.timeframe });
  if (!items.length) return null;
  return (
    <View style={styles.chipRow}>
      {items.map((it) => (
        <View key={it.label} style={styles.chip}>
          <Text style={styles.chipLabel}>{it.label}</Text>
          <Text style={styles.chipValue}>{it.value}</Text>
        </View>
      ))}
    </View>
  );
}

// ─── Variant card ─────────────────────────────────────────────────────────

function VariantCard({
  variant,
  rank,
  onApply,
  applying,
}: {
  variant: Variant;
  rank: number;
  onApply: () => void;
  applying: boolean;
}) {
  const stats = variant.stats || {};
  const closed = stats.closed_trades ?? 0;
  return (
    <View
      style={[
        styles.card,
        variant.improved ? styles.cardImproved : styles.cardWorse,
      ]}
    >
      <View style={styles.cardHeader}>
        <View style={{ flex: 1 }}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: spacing.xs }}>
            <Text style={styles.rankBadge}>#{rank}</Text>
            <Text style={styles.cardTitle} numberOfLines={1}>
              {variant.label}
            </Text>
          </View>
          <Text style={styles.cardTagline}>{variant.tagline}</Text>
        </View>
        <Pill
          label={variant.improved ? 'Improved' : 'Worse'}
          tone={variant.improved ? 'positive' : 'negative'}
        />
      </View>

      <PatchChips patch={variant.patch} />

      {/* Result row: PnL + delta vs baseline */}
      <View style={styles.statBlock}>
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Total P&L</Text>
          <Text style={[styles.statValue, { color: pnlTone(stats.total_pnl) === 'positive' ? colors.positive : pnlTone(stats.total_pnl) === 'negative' ? colors.negative : colors.text }]}>
            {fmtPnl(stats.total_pnl)}
          </Text>
          <Text
            style={[
              styles.statDelta,
              {
                color:
                  variant.delta_pnl > 0
                    ? colors.positive
                    : variant.delta_pnl < 0
                      ? colors.negative
                      : colors.textDim,
              },
            ]}
          >
            {fmtDelta(variant.delta_pnl)} vs current
          </Text>
        </View>
        <View style={styles.statCellDivider} />
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Win rate</Text>
          <Text style={styles.statValue}>
            {stats.win_rate !== undefined ? `${stats.win_rate.toFixed(1)}%` : '—'}
          </Text>
          <Text
            style={[
              styles.statDelta,
              {
                color:
                  variant.delta_win_rate > 0
                    ? colors.positive
                    : variant.delta_win_rate < 0
                      ? colors.negative
                      : colors.textDim,
              },
            ]}
          >
            {fmtDelta(variant.delta_win_rate, 'pp')}
          </Text>
        </View>
        <View style={styles.statCellDivider} />
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Max DD</Text>
          <Text style={styles.statValue}>
            {stats.max_drawdown !== undefined ? `${stats.max_drawdown.toFixed(1)}%` : '—'}
          </Text>
          <Text
            style={[
              styles.statDelta,
              {
                // For drawdown, smaller is better, so the colour logic is inverted.
                color:
                  variant.delta_dd < 0
                    ? colors.positive
                    : variant.delta_dd > 0
                      ? colors.negative
                      : colors.textDim,
              },
            ]}
          >
            {fmtDelta(variant.delta_dd)}
          </Text>
        </View>
      </View>

      <Text style={styles.tradeCount}>
        {closed} closed trade{closed === 1 ? '' : 's'} in this backtest
      </Text>

      <PrimaryButton
        label={applying ? 'Applying…' : 'Apply this tweak'}
        variant={variant.improved ? 'primary' : 'secondary'}
        onPress={onApply}
        disabled={applying}
        icon={
          applying ? (
            <ActivityIndicator size="small" color={colors.accentText} />
          ) : (
            <Ionicons
              name="checkmark-circle"
              size={16}
              color={variant.improved ? colors.accentText : colors.text}
            />
          )
        }
      />
    </View>
  );
}

// ─── Baseline card ────────────────────────────────────────────────────────

function BaselineCard({ stats }: { stats: BacktestStats }) {
  const closed = stats.closed_trades ?? 0;
  return (
    <View style={[styles.card, styles.cardBaseline]}>
      <View style={styles.cardHeader}>
        <View style={{ flex: 1 }}>
          <Text style={styles.cardTitle}>Your current settings</Text>
          <Text style={styles.cardTagline}>Baseline backtest</Text>
        </View>
        <Pill label="Baseline" tone="neutral" />
      </View>
      <View style={styles.statBlock}>
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Total P&L</Text>
          <Text style={[styles.statValue, { color: pnlTone(stats.total_pnl) === 'positive' ? colors.positive : pnlTone(stats.total_pnl) === 'negative' ? colors.negative : colors.text }]}>
            {fmtPnl(stats.total_pnl)}
          </Text>
        </View>
        <View style={styles.statCellDivider} />
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Win rate</Text>
          <Text style={styles.statValue}>
            {stats.win_rate !== undefined ? `${stats.win_rate.toFixed(1)}%` : '—'}
          </Text>
        </View>
        <View style={styles.statCellDivider} />
        <View style={styles.statCell}>
          <Text style={styles.statLabel}>Max DD</Text>
          <Text style={styles.statValue}>
            {stats.max_drawdown !== undefined ? `${stats.max_drawdown.toFixed(1)}%` : '—'}
          </Text>
        </View>
      </View>
      <Text style={styles.tradeCount}>
        {closed} closed trade{closed === 1 ? '' : 's'} in this backtest
      </Text>
    </View>
  );
}

// ─── Screen ───────────────────────────────────────────────────────────────

export default function OptimizeScreen() {
  const { id } = useLocalSearchParams<{ id: string }>();
  const sid = Number(id);
  const insets = useSafeAreaInsets();
  const router = useRouter();
  const { uid } = useAuth();
  const qc = useQueryClient();

  const [days, setDays] = useState<30 | 90>(30);
  const [result, setResult] = useState<OptimizeResult | null>(null);
  const [errorBlock, setErrorBlock] = useState<{ kind: 'pro' | 'timeout' | 'locked' | 'other'; msg: string } | null>(null);
  const [applyingLabel, setApplyingLabel] = useState<string | null>(null);

  // Run the optimizer (fan-out backtest sweep on the backend).
  const optimizeMut = useMutation({
    mutationFn: async () => {
      // Clear stale state at the start of every run so a failed rerun never
      // leaves the previous ranking visible alongside an error banner.
      setErrorBlock(null);
      setResult(null);
      const resp = await apiPostFlex<OptimizeResult & { error?: string; message?: string }>(
        `/api/strategies/${sid}/optimize`,
        { uid, days },
        uid,
      );
      if (resp.status === 402 || resp.body?.error === 'PRO_REQUIRED') {
        throw { kind: 'pro' as const, msg: resp.body?.message || 'Pro required' };
      }
      if (resp.status === 408 || resp.body?.error === 'TIMEOUT') {
        throw { kind: 'timeout' as const, msg: resp.body?.message || 'Timed out' };
      }
      if (resp.body?.error === 'LOCKED') {
        throw { kind: 'locked' as const, msg: resp.body?.message || 'Locked strategy' };
      }
      if (!resp.ok || !resp.body) {
        throw { kind: 'other' as const, msg: resp.body?.message || `Optimizer failed (${resp.status})` };
      }
      return resp.body as OptimizeResult;
    },
    onSuccess: (data) => {
      setResult(data);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    },
    onError: (err: { kind: string; msg: string }) => {
      setErrorBlock({ kind: (err.kind as any) || 'other', msg: err.msg || 'Something went wrong' });
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    },
  });

  // Apply a single variant's patch to the live strategy config.
  const applyMut = useMutation({
    mutationFn: async (variant: Variant) => {
      setApplyingLabel(variant.label);
      const resp = await apiPostFlex<ApplyResult & { error?: string; detail?: string }>(
        `/api/strategies/${sid}/optimize/apply`,
        { uid, patch: variant.patch },
        uid,
      );
      if (!resp.ok) {
        throw new Error(resp.body?.detail || 'Apply failed');
      }
      return resp.body as ApplyResult;
    },
    onSuccess: () => {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      // Bust strategy caches so the detail screen reflects the new params.
      qc.invalidateQueries({ queryKey: ['strategies', uid] });
      qc.invalidateQueries({ queryKey: ['strategy-detail', sid] });
      setApplyingLabel(null);
      Alert.alert(
        'Applied',
        'Your strategy now uses these settings. Past performance is not a guarantee — monitor it for a few days.',
        [{ text: 'Back to strategy', onPress: () => router.back() }],
      );
    },
    onError: (err: Error) => {
      setApplyingLabel(null);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      Alert.alert('Could not apply', err.message || 'Something went wrong');
    },
  });

  const handleApply = useCallback(
    (variant: Variant) => {
      if (applyingLabel) return;
      Alert.alert(
        `Apply "${variant.label}"?`,
        `${variant.tagline}\n\nThis updates your strategy in place. You can keep adjusting from the Builder later.`,
        [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Apply', onPress: () => applyMut.mutate(variant) },
        ],
      );
    },
    [applyMut, applyingLabel],
  );

  const isRunning = optimizeMut.isPending;

  return (
    <View style={[styles.screen, { paddingTop: insets.top }]}>
      <Stack.Screen options={{ headerShown: false }} />

      {/* Header */}
      <View style={styles.header}>
        <Pressable onPress={() => router.back()} hitSlop={12} style={styles.backBtn}>
          <Ionicons name="chevron-back" size={22} color={colors.text} />
        </Pressable>
        <View style={{ flex: 1 }}>
          <Text style={styles.headerTitle}>AI Strategy Tuner</Text>
          <Text style={styles.headerSub}>
            Tries ~9 parameter tweaks and ranks them by historical performance.
          </Text>
        </View>
      </View>

      <ScrollView
        contentContainerStyle={{ padding: spacing.lg, paddingBottom: insets.bottom + spacing.xxl }}
        showsVerticalScrollIndicator={false}
      >
        {/* Period + Run */}
        <View style={styles.controlsCard}>
          <Text style={styles.controlsLabel}>Backtest window</Text>
          <PeriodToggle value={days} onChange={setDays} disabled={isRunning} />
          <View style={{ height: spacing.md }} />
          <PrimaryButton
            label={isRunning ? 'Running optimizer…' : result ? 'Run again' : 'Find better settings'}
            onPress={() => optimizeMut.mutate()}
            disabled={isRunning}
            icon={
              isRunning ? (
                <ActivityIndicator size="small" color={colors.accentText} />
              ) : (
                <Ionicons name="sparkles" size={16} color={colors.accentText} />
              )
            }
          />
          <Text style={styles.disclaimer}>
            The tuner runs ~10 backtests in parallel. This typically takes 30–60 seconds.
            Past performance does not guarantee future results.
          </Text>
        </View>

        {/* Loading state */}
        {isRunning && !result && (
          <View style={[styles.card, { alignItems: 'center', paddingVertical: spacing.xxl }]}>
            <ActivityIndicator size="large" color={colors.accent} />
            <Text style={[styles.cardTitle, { marginTop: spacing.md }]}>
              Running ~10 backtests…
            </Text>
            <Text style={styles.cardTagline}>
              Trying tighter stops, wider targets, leverage tweaks, and timeframe shifts.
            </Text>
          </View>
        )}

        {/* Errors */}
        {errorBlock && !isRunning && (
          <View style={{ marginTop: spacing.md }}>
            <EmptyState
              icon={errorBlock.kind === 'pro' ? 'lock-closed' : errorBlock.kind === 'timeout' ? 'time' : 'alert-circle'}
              title={
                errorBlock.kind === 'pro'
                  ? 'Pro subscription required'
                  : errorBlock.kind === 'timeout'
                    ? 'Optimizer timed out'
                    : errorBlock.kind === 'locked'
                      ? 'Locked strategy'
                      : 'Optimizer failed'
              }
              hint={errorBlock.msg}
            />
          </View>
        )}

        {/* Results */}
        {result && !isRunning && (
          <>
            <Text style={styles.sectionLabel}>Baseline</Text>
            <BaselineCard stats={result.baseline.stats} />

            <Text style={[styles.sectionLabel, { marginTop: spacing.lg }]}>
              Suggested tweaks ({result.variants.length})
            </Text>
            {result.variants.length === 0 ? (
              <EmptyState
                icon="alert-circle"
                title="Couldn't generate variants"
                hint="None of the parameter tweaks produced a usable backtest. Try the 90-day window."
              />
            ) : !result.any_improved ? (
              <View style={[styles.card, { borderColor: colors.warning }]}>
                <Text style={[styles.cardTitle, { color: colors.warning }]}>
                  Already well-tuned
                </Text>
                <Text style={styles.cardTagline}>
                  Every tweak scored worse than your current settings on this window.
                  The closest variants are listed below — you can still try one if you
                  want to experiment.
                </Text>
              </View>
            ) : null}
            {result.variants.length > 0 && (
              result.variants.map((v, i) => (
                <VariantCard
                  key={v.label}
                  variant={v}
                  rank={i + 1}
                  onApply={() => handleApply(v)}
                  applying={applyingLabel === v.label}
                />
              ))
            )}
          </>
        )}
      </ScrollView>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },

  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.sm,
    paddingBottom: spacing.md,
    gap: spacing.sm,
  },
  backBtn: {
    width: 32, height: 32, borderRadius: 16,
    alignItems: 'center', justifyContent: 'center',
    backgroundColor: colors.card,
    marginTop: 2,
  },
  headerTitle: { color: colors.text, fontSize: 20, fontFamily: font.bold },
  headerSub:   { color: colors.textDim, fontSize: 13, fontFamily: font.regular, marginTop: 2 },

  controlsCard: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 1, borderColor: colors.border,
    marginBottom: spacing.lg,
  },
  controlsLabel: {
    color: colors.textDim, fontSize: 12, fontFamily: font.medium,
    textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: spacing.sm,
  },
  disclaimer: {
    color: colors.textMute, fontSize: 11, fontFamily: font.regular,
    marginTop: spacing.sm, lineHeight: 16,
  },

  toggleWrap: {
    flexDirection: 'row',
    backgroundColor: colors.bgElev,
    borderRadius: radius.md,
    padding: 3,
    gap: 3,
  },
  toggleBtn: {
    flex: 1, paddingVertical: 10,
    alignItems: 'center', borderRadius: radius.sm,
  },
  toggleBtnActive: { backgroundColor: colors.cardHi },
  toggleTxt:       { color: colors.textDim, fontSize: 13, fontFamily: font.medium },
  toggleTxtActive: { color: colors.text, fontFamily: font.semibold },

  sectionLabel: {
    color: colors.textDim, fontSize: 12, fontFamily: font.medium,
    textTransform: 'uppercase', letterSpacing: 0.5,
    marginBottom: spacing.sm,
  },

  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    padding: spacing.lg,
    borderWidth: 1, borderColor: colors.border,
    marginBottom: spacing.md,
  },
  cardImproved: { borderColor: colors.positive },
  cardWorse:    { borderColor: colors.border, opacity: 0.85 },
  cardBaseline: { borderColor: colors.borderHi },

  cardHeader: {
    flexDirection: 'row', alignItems: 'flex-start',
    gap: spacing.sm, marginBottom: spacing.md,
  },
  rankBadge: {
    color: colors.textDim, fontSize: 11, fontFamily: font.bold,
    backgroundColor: colors.cardHi,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 4,
  },
  cardTitle:   { color: colors.text, fontSize: 16, fontFamily: font.semibold, flexShrink: 1 },
  cardTagline: { color: colors.textDim, fontSize: 12, fontFamily: font.regular, marginTop: 2 },

  chipRow: {
    flexDirection: 'row', flexWrap: 'wrap',
    gap: spacing.xs, marginBottom: spacing.md,
  },
  chip: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: colors.bgElev,
    borderRadius: radius.sm,
    paddingHorizontal: 8, paddingVertical: 4,
    gap: 4,
  },
  chipLabel: { color: colors.textMute, fontSize: 10, fontFamily: font.medium, textTransform: 'uppercase' },
  chipValue: { color: colors.text, fontSize: 12, fontFamily: font.semibold },

  statBlock: {
    flexDirection: 'row', alignItems: 'stretch',
    backgroundColor: colors.bgElev,
    borderRadius: radius.md,
    paddingVertical: spacing.sm,
    marginBottom: spacing.sm,
  },
  statCell: { flex: 1, alignItems: 'center', paddingHorizontal: 4 },
  statCellDivider: { width: 1, backgroundColor: colors.border },
  statLabel: { color: colors.textMute, fontSize: 10, fontFamily: font.medium, textTransform: 'uppercase' },
  statValue: { color: colors.text, fontSize: 16, fontFamily: font.bold, marginTop: 2 },
  statDelta: { fontSize: 11, fontFamily: font.medium, marginTop: 2 },

  tradeCount: {
    color: colors.textMute, fontSize: 11, fontFamily: font.regular,
    textAlign: 'center', marginBottom: spacing.sm,
  },
});
