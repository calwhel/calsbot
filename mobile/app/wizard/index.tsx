/**
 * TradeHub mobile — Strategy Builder Wizard (v1.1).
 *
 * 7-step flow that mirrors the web /portal builder:
 *   1. Style preset
 *   2. Direction & build mode (paper / live)
 *   3. Primary entry signal (15-type library)
 *   4. Confirmations (0-3 additional signals)
 *   5. Exit targets (TP1 / TP2 / SL / trailing / breakeven)
 *   6. Risk panel + universe + sessions/days + filters
 *   7. Review — name, AI suggestions, in-flow backtest, save & publish
 *
 * Builds the /api/save-strategy payload deterministically from wizard state
 * (mobile/lib/wizardConfig.ts), no AI compile step required, so the wizard
 * works for free users too. Pro-only extras (Chat / Pine / Scanner / AI
 * Indicator Generator) are deferred to a separate v1.2 task.
 */

import React, { useCallback, useMemo, useRef, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TextInput, Pressable,
  KeyboardAvoidingView, Platform, ActivityIndicator,
  LayoutAnimation, UIManager, Animated as RNAnimated,
} from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';

if (Platform.OS === 'android' && UIManager.setLayoutAnimationEnabledExperimental) {
  UIManager.setLayoutAnimationEnabledExperimental(true);
}

import { PrimaryButton } from '@/components/PrimaryButton';
import { Pill } from '@/components/Pill';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiPost, apiPostFlex, type SaveStrategyResponse, type BacktestResult } from '@/lib/api';

import {
  STYLE_LABELS, STYLE_PRESETS, STYLE_SIGNALS, SIGNAL_META,
  SESSIONS, DAYS, getDefaultCfg, estimateFireRate, calcRiskLevel,
  type StyleId, type SignalType, type Tf, type Direction, type CoinUniverse,
  type RiskProfile, type BtcRegime, type Session, type Day,
} from '@/lib/strategyPresets';
import {
  buildWizardConfig, defaultWizardState, getWizardWarnings, validateStep,
  type WizardState, type Confirm,
} from '@/lib/wizardConfig';

import { StyleCard } from '@/components/wizard/StyleCard';
import { ChipRow, type ChipOption } from '@/components/wizard/ChipRow';
import { Stepper } from '@/components/wizard/Stepper';
import { SectionHeader } from '@/components/wizard/SectionHeader';
import { ConditionPicker } from '@/components/wizard/ConditionPicker';
import { ConditionEditor } from '@/components/wizard/ConditionEditor';

const WZ_STEPS = [
  { key: 'style',   label: 'Style',     icon: '🎨' },
  { key: 'dir',     label: 'Direction', icon: '↕️' },
  { key: 'signal',  label: 'Signal',    icon: '⚡' },
  { key: 'confirm', label: 'Confirm',   icon: '🔍' },
  { key: 'exit',    label: 'Exit',      icon: '🛡️' },
  { key: 'risk',    label: 'Risk',      icon: '⚙️' },
  { key: 'review',  label: 'Launch',    icon: '🚀' },
] as const;

const DIR_OPTIONS: ChipOption<Direction>[] = [
  { value: 'LONG',  icon: '📈', label: 'Long only' },
  { value: 'SHORT', icon: '📉', label: 'Short only' },
  { value: 'BOTH',  icon: '↕️', label: 'Both ways' },
];
const MODE_OPTIONS: ChipOption<'paper' | 'live'>[] = [
  { value: 'paper', icon: '🧪', label: 'Paper trade' },
  { value: 'live',  icon: '⚡', label: 'Live trade' },
];
const TF_OPTIONS: ChipOption<Tf>[] = [
  { value: '1m',  label: '1m' }, { value: '3m',  label: '3m' },
  { value: '5m',  label: '5m' }, { value: '15m', label: '15m' },
  { value: '30m', label: '30m'},  { value: '1h',  label: '1h' },
  { value: '4h',  label: '4h' }, { value: '1d',  label: '1d' },
];
const COIN_OPTIONS: ChipOption<CoinUniverse>[] = [
  { value: 'all',      icon: '🌐', label: 'All coins' },
  { value: 'gainers',  icon: '📈', label: 'Top gainers' },
  { value: 'losers',   icon: '📉', label: 'Top losers' },
  { value: 'specific', icon: '📋', label: 'Watchlist' },
  { value: 'single',   icon: '🎯', label: 'Single coin' },
];
const RISK_PROFILE_OPTIONS: ChipOption<RiskProfile>[] = [
  { value: 'low',    icon: '🟢', label: 'Low (sniper)' },
  { value: 'medium', icon: '🟡', label: 'Medium' },
  { value: 'high',   icon: '🔴', label: 'High (loose)' },
];
const BTC_REGIME_OPTIONS: ChipOption<BtcRegime>[] = [
  { value: 'any',     label: 'Any regime' },
  { value: 'bullish', icon: '🟢', label: 'Bullish only' },
  { value: 'bearish', icon: '🔴', label: 'Bearish only' },
  { value: 'neutral', icon: '⚪', label: 'Neutral only' },
];
type NameSuggestion = { name: string; tagline: string };
type SaveResult = { id: number; name: string; status: string };

export default function WizardScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();

  const [s, setS] = useState<WizardState>(defaultWizardState);
  const scrollRef = useRef<ScrollView | null>(null);

  // ── Transient UI state ─────────────────────────────────────────────────
  const [pickerVisible, setPickerVisible]       = useState<null | 'primary' | 'confirm'>(null);
  const [editingConfirmIdx, setEditingConfirmIdx] = useState<number | null>(null);
  const [stepError, setStepError]               = useState<string | null>(null);

  const [nameSuggestions, setNameSuggestions]   = useState<NameSuggestion[]>([]);
  const [nameLoading, setNameLoading]           = useState(false);

  const [btResult, setBtResult] = useState<BacktestResult | null>(null);
  const [btLoading, setBtLoading] = useState(false);
  const [btError, setBtError]   = useState<string | null>(null);

  const [saveResult, setSaveResult] = useState<SaveResult | null>(null);
  const [publishLoading, setPublishLoading] = useState(false);
  const [publishDone, setPublishDone] = useState(false);

  const update = useCallback((patch: Partial<WizardState>) => {
    setS(prev => ({ ...prev, ...patch }));
    setStepError(null);
  }, []);

  const goTo = (n: number) => {
    setStepError(null);
    LayoutAnimation.configureNext(LayoutAnimation.create(280, 'easeInEaseOut', 'opacity'));
    setS(prev => ({ ...prev, step: Math.max(1, Math.min(7, n)) }));
    requestAnimationFrame(() => scrollRef.current?.scrollTo({ y: 0, animated: true }));
  };

  // ── Style preset application — mirrors web setWzStyle() ────────────────
  const applyStyle = (style: StyleId) => {
    const p = STYLE_PRESETS[style];
    const recommended = STYLE_SIGNALS[style][0];
    setS(prev => ({
      ...prev,
      style,
      timeframe: p.timeframe,
      tp1: p.tp1, sl: p.sl,
      leverage: p.leverage,
      posSize: p.posSize,
      maxPos: p.maxPos,
      maxTrades: p.maxTrades,
      cooldown: p.cooldown,
      dailyLoss: p.dailyLoss,
      // Preselect a sensible primary signal if none chosen yet
      primaryType: prev.primaryType ?? recommended,
      primaryCfg: prev.primaryType ? prev.primaryCfg : getDefaultCfg(recommended, p.timeframe),
    }));
  };

  // ── Confirmation list helpers ──────────────────────────────────────────
  const addConfirm = (type: SignalType) => {
    const cfg = getDefaultCfg(type, s.timeframe);
    setS(prev => ({ ...prev, confirms: [...prev.confirms, { type, cfg }] }));
  };
  const updateConfirm = (idx: number, cfg: Record<string, any>) => {
    setS(prev => ({
      ...prev,
      confirms: prev.confirms.map((c, i) => i === idx ? { ...c, cfg } : c),
    }));
  };
  const removeConfirm = (idx: number) => {
    if (Platform.OS !== 'web') Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(() => {});
    setS(prev => ({ ...prev, confirms: prev.confirms.filter((_, i) => i !== idx) }));
  };

  // ── Validation + navigation ────────────────────────────────────────────
  const onNext = () => {
    const err = validateStep(s.step, s);
    if (err) {
      setStepError(err);
      if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
      return;
    }
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    goTo(s.step + 1);
  };
  const onBack = () => goTo(s.step - 1);

  // ── AI name suggestion ────────────────────────────────────────────────
  const suggestNames = async () => {
    if (!uid) return;
    setNameLoading(true);
    try {
      const r = await apiPostFlex<{ names?: NameSuggestion[]; error?: string }>(
        '/api/wizard/suggest-name',
        {
          uid,
          style: s.style || 'general',
          direction: s.direction,
          primaryType: s.primaryType || 'rsi',
          confirms: s.confirms.map(c => ({ type: c.type })),
          tp1: s.tp1, sl: s.sl, leverage: s.leverage,
          coins: s.coins, timeframe: s.timeframe,
        },
      );
      if (r.ok && r.body.names?.length) {
        setNameSuggestions(r.body.names);
      } else {
        setNameSuggestions([]);
        setStepError(r.body.error || 'Could not generate names — please try again.');
      }
    } finally {
      setNameLoading(false);
    }
  };

  // ── In-flow backtest (Pro path on web; mobile shows paywall on 402) ────
  const runBacktest = async (days: 30 | 90) => {
    if (!uid) return;
    setBtLoading(true); setBtError(null); setBtResult(null);
    try {
      const r = await apiPostFlex<BacktestResult>('/api/backtest/run', {
        uid, days,
        config: {
          direction: s.direction,
          primaryType: s.primaryType,
          primaryCfg:  s.primaryCfg || {},
          confirms:    s.confirms,
          tp1: s.tp1, sl: s.sl, leverage: s.leverage,
          timeframe: s.timeframe,
          singleCoin: s.coins === 'single' && s.singleCoin ? s.singleCoin : 'BTCUSDT',
        },
      });
      if (r.status === 402) {
        setBtError('Backtest is a Pro feature. Upgrade in the web app to unlock it.');
      } else if (r.status === 0) {
        setBtError('Network error — check your connection and try again.');
      } else if (!r.ok || r.body.error) {
        setBtError(r.body.message || 'Backtest failed — try a shorter window.');
      } else {
        setBtResult(r.body);
      }
    } finally {
      setBtLoading(false);
    }
  };

  // ── Save strategy ─────────────────────────────────────────────────────
  const saveMutation = useMutation({
    mutationFn: async () => {
      const cfg = buildWizardConfig(s);
      const res = await apiPost<SaveStrategyResponse>('/api/save-strategy', { uid, config: cfg });
      return res;
    },
    onSuccess: (data) => {
      setSaveResult({ id: data.id, name: data.name, status: data.status });
      qc.invalidateQueries({ queryKey: ['strategies'] });
      qc.invalidateQueries({ queryKey: ['portfolio'] });
      if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
    },
    onError: (err: any) => {
      setStepError(err?.message || 'Save failed — please try again.');
    },
  });

  const publishToMarketplace = async () => {
    if (!saveResult || !uid) return;
    setPublishLoading(true);
    try {
      const r = await apiPostFlex<{ success?: boolean; listing_id?: number; detail?: string }>(
        `/api/strategies/${saveResult.id}/share`, {}, uid, { uid },
      );
      if (r.ok && r.body.success) {
        setPublishDone(true);
        qc.invalidateQueries({ queryKey: ['marketplace'] });
        if (Platform.OS !== 'web') Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      } else {
        setStepError(r.body.detail || 'Publish failed — please try again.');
      }
    } finally {
      setPublishLoading(false);
    }
  };

  // ── Live preview line shown under the dots ─────────────────────────────
  const summary = useMemo(() => {
    if (s.step === 1 || !s.style) return 'Build your perfect strategy';
    const dir = { LONG: '📈 Long', SHORT: '📉 Short', BOTH: '↕ Both' }[s.direction];
    const sig = s.primaryType ? `${SIGNAL_META[s.primaryType].icon} ${SIGNAL_META[s.primaryType].label}` : '— signal';
    const coin = s.coins === 'single' && s.singleCoin ? ` · 🎯 ${s.singleCoin.replace(/USDT$/i, '')}`
               : s.coins === 'gainers' ? ' · 📈 gainers'
               : s.coins === 'losers'  ? ' · 📉 losers'
               : '';
    return `${s.style} · ${dir} · ${sig} · ${s.timeframe} · TP ${s.tp1}% / SL ${s.sl}% · ${s.leverage}×${coin}`;
  }, [s]);

  const warnings = useMemo(() => getWizardWarnings(s), [s]);
  const risk = calcRiskLevel(s.leverage);
  const fireRate = estimateFireRate(s.primaryType, s.timeframe);

  // ──────────────────────────────────────────────────────────────────────
  // Render
  // ──────────────────────────────────────────────────────────────────────
  return (
    <View style={styles.root}>
      <Stack.Screen
        options={{
          title: 'Strategy Builder',
          headerStyle: { backgroundColor: colors.bg },
          headerTitleStyle: { color: colors.text, fontFamily: font.bold, fontSize: 16 },
          headerTintColor: colors.text,
          headerBackTitle: 'Back',
        }}
      />

      <View style={styles.progressWrap}>
        <View style={styles.progressTrack}>
          <View style={[styles.progressFill, { width: `${((s.step - 1) / 6) * 100}%` }]} />
        </View>
        <View style={styles.dotsRow}>
          {WZ_STEPS.map((step, i) => {
            const idx = i + 1;
            const done = idx < s.step;
            const active = idx === s.step;
            return (
              <Pressable
                key={step.key}
                onPress={() => done && goTo(idx)}
                hitSlop={6}
                style={styles.dotWrap}
                accessibilityRole={done ? 'button' : undefined}
                accessibilityLabel={`Step ${idx}: ${step.label}${active ? ' (current)' : done ? ' (completed)' : ''}`}
              >
                <View style={[styles.dot, done && styles.dotDone, active && styles.dotActive]}>
                  <Text style={[styles.dotIcon, (done || active) && styles.dotIconActive]}>
                    {done ? '✓' : step.icon}
                  </Text>
                </View>
                <Text
                  style={[styles.dotLabel, active && styles.dotLabelActive]}
                  numberOfLines={1}
                >
                  {step.label}
                </Text>
              </Pressable>
            );
          })}
        </View>
      </View>

      <View style={styles.summaryBar}>
        <Text style={styles.summaryStep}>Step {s.step} of 7</Text>
        <Text style={styles.summary} numberOfLines={1}>{summary}</Text>
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={insets.top + 60}
        style={{ flex: 1 }}
      >
        <ScrollView
          ref={scrollRef}
          style={{ flex: 1 }}
          contentContainerStyle={[styles.body, { paddingBottom: 140 + insets.bottom }]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {s.step === 1 && <Step1 s={s} onPick={applyStyle} />}
          {s.step === 2 && <Step2 s={s} update={update} />}
          {s.step === 3 && (
            <Step3
              s={s}
              update={update}
              onPickPrimary={() => setPickerVisible('primary')}
            />
          )}
          {s.step === 4 && (
            <Step4
              s={s}
              onAdd={() => setPickerVisible('confirm')}
              onUpdate={updateConfirm}
              onRemove={removeConfirm}
              editingIdx={editingConfirmIdx}
              setEditingIdx={setEditingConfirmIdx}
            />
          )}
          {s.step === 5 && <Step5 s={s} update={update} warnings={warnings} risk={risk} />}
          {s.step === 6 && <Step6 s={s} update={update} />}
          {s.step === 7 && (
            <Step7
              s={s}
              update={update}
              fireRate={fireRate}
              warnings={warnings}
              risk={risk}
              nameSuggestions={nameSuggestions}
              nameLoading={nameLoading}
              onSuggestNames={suggestNames}
              btResult={btResult}
              btLoading={btLoading}
              btError={btError}
              onRunBacktest={runBacktest}
              saveResult={saveResult}
              saving={saveMutation.isPending}
              onSave={() => saveMutation.mutate()}
              publishLoading={publishLoading}
              publishDone={publishDone}
              onPublish={publishToMarketplace}
              onDone={() => router.replace('/(tabs)/strategies' as any)}
            />
          )}

          {stepError ? (
            <View style={styles.errorBox}>
              <Ionicons name="alert-circle" size={16} color={colors.negative} />
              <Text style={styles.errorTxt}>{stepError}</Text>
            </View>
          ) : null}
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Sticky footer ─ Back / Next buttons */}
      {!saveResult || s.step !== 7 ? (
        <View style={[styles.footer, { paddingBottom: Math.max(insets.bottom, 12) }]}>
          {s.step > 1 ? (
            <Pressable onPress={onBack} style={styles.backBtn} accessibilityRole="button">
              <Ionicons name="chevron-back" size={18} color={colors.text} />
              <Text style={styles.backTxt}>Back</Text>
            </Pressable>
          ) : <View style={{ width: 88 }} />}
          {s.step < 7 ? (
            <PrimaryButton
              label={s.step === 6 ? 'Review →' : 'Next →'}
              onPress={onNext}
            />
          ) : (
            <PrimaryButton
              label={saveMutation.isPending ? 'Saving…' : '✅ Save Strategy'}
              onPress={() => saveMutation.mutate()}
              loading={saveMutation.isPending}
              disabled={!s.name.trim() || !s.primaryType}
            />
          )}
        </View>
      ) : null}

      <ConditionPicker
        visible={pickerVisible !== null}
        onClose={() => setPickerVisible(null)}
        current={pickerVisible === 'primary' ? s.primaryType : null}
        style={s.style}
        title={pickerVisible === 'primary' ? 'Pick your entry signal' : 'Add a confirmation'}
        excludeTypes={pickerVisible === 'confirm'
          ? [...s.confirms.map(c => c.type), ...(s.primaryType ? [s.primaryType] : [])]
          : []}
        onPick={(t) => {
          if (pickerVisible === 'primary') {
            update({ primaryType: t, primaryCfg: getDefaultCfg(t, s.timeframe) });
          } else {
            addConfirm(t);
          }
        }}
      />
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 1 — Trading Style
// ─────────────────────────────────────────────────────────────────────────
function Step1({ s, onPick }: { s: WizardState; onPick: (id: StyleId) => void }) {
  return (
    <View>
      <StepIntro
        title="What kind of trader are you?"
        subtitle="Each style picks sensible defaults for leverage, TP/SL, and timeframe — you can change anything in the next steps."
      />
      <View style={styles.styleGrid}>
        {(Object.keys(STYLE_LABELS) as StyleId[]).map(id => (
          <StyleCard
            key={id}
            icon={STYLE_LABELS[id].icon}
            label={STYLE_LABELS[id].label}
            tagline={STYLE_LABELS[id].tagline}
            selected={s.style === id}
            onPress={() => onPick(id)}
          />
        ))}
      </View>
      {s.style ? (
        <View style={styles.previewCard}>
          <Text style={styles.previewTitle}>{STYLE_LABELS[s.style].icon} {STYLE_LABELS[s.style].label} defaults</Text>
          <Text style={styles.previewLine}>
            Timeframe <Text style={styles.previewVal}>{STYLE_PRESETS[s.style].timeframe}</Text> ·
            TP <Text style={styles.previewVal}>{STYLE_PRESETS[s.style].tp1}%</Text> /
            SL <Text style={styles.previewVal}>{STYLE_PRESETS[s.style].sl}%</Text> ·
            Lev <Text style={styles.previewVal}>{STYLE_PRESETS[s.style].leverage}×</Text>
          </Text>
        </View>
      ) : null}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 2 — Direction & Mode
// ─────────────────────────────────────────────────────────────────────────
function Step2({ s, update }: { s: WizardState; update: (p: Partial<WizardState>) => void }) {
  return (
    <View>
      <StepIntro
        title="Which direction & mode?"
        subtitle="Long catches uptrends, short profits from drops, both adapts. Paper-trade to validate before risking funds."
      />
      <Card>
        <SectionHeader label="Trade direction" icon="↕️" />
        <ChipRow options={DIR_OPTIONS} value={s.direction} onChange={(v) => update({ direction: v })} />
      </Card>
      <Card>
        <SectionHeader label="Build mode" icon="🧪" />
        <ChipRow options={MODE_OPTIONS} value={s.mode} onChange={(v) => update({ mode: v })} />
        <Text style={styles.hint}>
          {s.mode === 'paper'
            ? '🧪 Paper mode tracks signals without sending real orders — perfect for testing.'
            : '⚡ Live mode places real Bitunix orders. Make sure you trust this strategy.'}
        </Text>
      </Card>
      <Card>
        <SectionHeader label="Candle timeframe" icon="🕐" hint="The default candle for all signals — each signal can override below." />
        <ChipRow options={TF_OPTIONS} value={s.timeframe} onChange={(v) => update({ timeframe: v })} />
      </Card>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 3 — Primary Entry Signal
// ─────────────────────────────────────────────────────────────────────────
function Step3({
  s, update, onPickPrimary,
}: {
  s: WizardState; update: (p: Partial<WizardState>) => void; onPickPrimary: () => void;
}) {
  const fireRate = estimateFireRate(s.primaryType, s.timeframe);
  return (
    <View>
      <StepIntro
        title="What triggers the trade?"
        subtitle="The primary signal is the spark — when this fires, the strategy considers entering. You can add confirmations in the next step."
      />
      <Card>
        <SectionHeader label="Entry signal" />
        <Pressable
          onPress={onPickPrimary}
          style={({ pressed }) => [styles.signalPickRow, pressed && { opacity: 0.85 }]}
          accessibilityRole="button"
        >
          <Text style={styles.signalIcon}>
            {s.primaryType ? SIGNAL_META[s.primaryType].icon : '➕'}
          </Text>
          <View style={{ flex: 1, minWidth: 0 }}>
            <Text style={styles.signalLabel}>
              {s.primaryType ? SIGNAL_META[s.primaryType].label : 'Pick a signal…'}
            </Text>
            <Text style={styles.signalDesc} numberOfLines={1}>
              {s.primaryType
                ? SIGNAL_META[s.primaryType].desc
                : 'Browse the 15-signal library'}
            </Text>
          </View>
          <Ionicons name="chevron-forward" size={18} color={colors.textDim} />
        </Pressable>
      </Card>

      {s.primaryType ? (
        <Card>
          <SectionHeader label="Configure signal" />
          <ConditionEditor
            type={s.primaryType}
            cfg={s.primaryCfg}
            onChange={(cfg) => update({ primaryCfg: cfg })}
          />
          {fireRate ? (
            <View style={styles.fireRow}>
              <Pill label={`🔥 Fires ${fireRate}`} tone="accent" small />
            </View>
          ) : null}
        </Card>
      ) : null}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 4 — Confirmations
// ─────────────────────────────────────────────────────────────────────────
function Step4({
  s, onAdd, onUpdate, onRemove, editingIdx, setEditingIdx,
}: {
  s: WizardState;
  onAdd: () => void;
  onUpdate: (idx: number, cfg: Record<string, any>) => void;
  onRemove: (idx: number) => void;
  editingIdx: number | null;
  setEditingIdx: (n: number | null) => void;
}) {
  const max = 3;
  return (
    <View>
      <StepIntro
        title="Add confirmations (optional)"
        subtitle="Confirmations make your strategy stricter — fewer trades, but higher quality. We recommend 1–2 confirmations."
      />
      <Card>
        <View style={styles.confirmsHeader}>
          <SectionHeader label={`Confirmations (${s.confirms.length}/${max})`} />
          {s.confirms.length < max ? (
            <Pressable
              onPress={onAdd}
              style={({ pressed }) => [styles.addBtn, pressed && { opacity: 0.85 }]}
              accessibilityRole="button"
            >
              <Ionicons name="add" size={16} color={colors.accent} />
              <Text style={styles.addBtnTxt}>Add</Text>
            </Pressable>
          ) : null}
        </View>

        {s.confirms.length === 0 ? (
          <View style={styles.emptyBox}>
            <Text style={styles.emptyTitle}>No confirmations yet</Text>
            <Text style={styles.emptyHint}>
              You can save a strategy with just the primary signal — but adding 1–2 confirmations dramatically improves win rate.
            </Text>
          </View>
        ) : null}

        {s.confirms.map((c, i) => {
          const meta = SIGNAL_META[c.type];
          const expanded = editingIdx === i;
          return (
            <View key={`${c.type}-${i}`} style={styles.confirmRow}>
              <Pressable
                onPress={() => setEditingIdx(expanded ? null : i)}
                style={styles.confirmHead}
                accessibilityRole="button"
                accessibilityLabel={`Toggle config for ${meta.label}`}
              >
                <Text style={styles.confirmIcon}>{meta.icon}</Text>
                <View style={{ flex: 1, minWidth: 0 }}>
                  <Text style={styles.confirmLabel}>{meta.label}</Text>
                  <Text style={styles.confirmDesc} numberOfLines={1}>
                    {summarizeCfg(c)}
                  </Text>
                </View>
                <Pressable
                  onPress={() => onRemove(i)}
                  hitSlop={10}
                  style={styles.removeBtn}
                  accessibilityLabel={`Remove ${meta.label}`}
                >
                  <Ionicons name="close" size={14} color={colors.negative} />
                </Pressable>
                <Ionicons
                  name={expanded ? 'chevron-up' : 'chevron-down'}
                  size={16}
                  color={colors.textDim}
                  style={{ marginLeft: 4 }}
                />
              </Pressable>
              {expanded ? (
                <View style={styles.confirmBody}>
                  <ConditionEditor
                    type={c.type}
                    cfg={c.cfg}
                    onChange={(cfg) => onUpdate(i, cfg)}
                    compact
                  />
                </View>
              ) : null}
            </View>
          );
        })}
      </Card>
    </View>
  );
}

function summarizeCfg(c: Confirm): string {
  const cfg = c.cfg;
  const tf = cfg.timeframe || '15m';
  switch (c.type) {
    case 'rsi':                return `${cfg.condition === 'lt' ? 'Oversold' : cfg.condition === 'gt' ? 'Overbought' : (cfg.condition || 'lt').replace(/_/g,' ')} · ${tf}`;
    case 'macd':               return `${(cfg.condition || 'bullish_cross').replace(/_/g,' ')} · ${tf}`;
    case 'ema':                return `${cfg.periods || '9/21'} · ${(cfg.condition || 'golden_cross').replace(/_/g,' ')}`;
    case 'volume_spike':       return `${cfg.multiplier || 2}× spike · ${tf}`;
    case 'price_momentum':     return `${cfg.pm_dir || 'up'} ${cfg.pm_pct || 10}% in ${cfg.pm_window || 15}m`;
    case 'breakout':           return `${cfg.bo_dir || 'up'} · ${cfg.bo_lookback || 20}c · ${cfg.bo_pct || 1}%`;
    case 'fvg':                return `${cfg.fvg_dir || 'bullish'} · ${(cfg.condition || 'gap_exists').replace(/_/g,' ')}`;
    case 'order_block':        return `${cfg.ob_type || 'bullish'} · ${tf}`;
    case 'vwap_deviation':     return `${cfg.vwap_pct || 3}% ${cfg.vwap_side || 'below'} VWAP`;
    case 'support_resistance': return `${(cfg.condition || 'at_support').replace(/_/g,' ')} · ${tf}`;
    case 'candlestick':        return `${(cfg.pattern || 'bullish_engulfing').replace(/_/g,' ')} · ${tf}`;
    case 'market_structure':   return `${(cfg.condition || 'bos_bullish').replace(/_/g,' ')} · ${tf}`;
    case 'adx_filter':         return `${cfg.condition || 'ranging'} · ${tf}`;
    case 'bb':                 return `${(cfg.condition || 'squeeze').replace(/_/g,' ')} · ${tf}`;
    case 'stoch_rsi':          return `${(cfg.condition || 'oversold').replace(/_/g,' ')} · ${tf}`;
    default: return tf;
  }
}

// ─────────────────────────────────────────────────────────────────────────
// Step 5 — Exit targets
// ─────────────────────────────────────────────────────────────────────────
function Step5({
  s, update, warnings, risk,
}: {
  s: WizardState;
  update: (p: Partial<WizardState>) => void;
  warnings: ReturnType<typeof getWizardWarnings>;
  risk: ReturnType<typeof calcRiskLevel>;
}) {
  const rr = (s.tp1 / s.sl).toFixed(1);
  return (
    <View>
      <StepIntro
        title="When do you exit?"
        subtitle="Set the price targets that close the position. Optional TP2 lets the trade run further; trailing locks profits as price moves."
      />
      <Card>
        <SectionHeader label="Take profit & stop loss" />
        <Stepper
          label="Take Profit 1"
          value={s.tp1}
          onChange={(v) => update({ tp1: v })}
          min={0.5} max={50} step={0.5} unit="%" decimals={1}
          presets={[1, 2, 3, 5, 8, 12]}
          hint="Where you take profit on most of the position"
        />
        <Stepper
          label="Stop Loss"
          value={s.sl}
          onChange={(v) => update({ sl: v })}
          min={0.3} max={20} step={0.3} unit="%" decimals={1}
          presets={[0.5, 1, 1.5, 2, 3, 5]}
          hint="Maximum loss before the trade is closed"
        />
        <View style={styles.rrRow}>
          <Pill
            label={`R:R ${rr}:1`}
            tone={Number(rr) >= 2 ? 'positive' : Number(rr) >= 1.5 ? 'accent' : Number(rr) >= 1 ? 'warning' : 'negative'}
            small
          />
          <Text style={styles.rrHint}>
            {Number(rr) >= 2 ? 'Strong reward-to-risk' : Number(rr) >= 1.5 ? 'Healthy reward-to-risk' : 'Consider widening TP'}
          </Text>
        </View>

        <View style={styles.rrBarWrap}>
          <View style={styles.rrBarLabel}>
            <Text style={[styles.rrBarTxt, { color: colors.negative }]}>SL {s.sl}%</Text>
            <Text style={[styles.rrBarTxt, { color: colors.textDim }]}>Entry</Text>
            <Text style={[styles.rrBarTxt, { color: colors.positive }]}>TP1 {s.tp1}%</Text>
            {s.tp2 != null ? <Text style={[styles.rrBarTxt, { color: colors.accent }]}>TP2 {s.tp2}%</Text> : null}
          </View>
          <View style={styles.rrBarTrack}>
            <View style={[styles.rrBarSeg, { flex: s.sl, backgroundColor: colors.negativeDim, borderTopLeftRadius: 4, borderBottomLeftRadius: 4 }]}>
              <View style={[styles.rrBarFill, { backgroundColor: colors.negative }]} />
            </View>
            <View style={[styles.rrBarEntry, { backgroundColor: colors.text }]} />
            <View style={[styles.rrBarSeg, { flex: s.tp1, backgroundColor: colors.positiveDim, borderTopRightRadius: s.tp2 ? 0 : 4, borderBottomRightRadius: s.tp2 ? 0 : 4 }]}>
              <View style={[styles.rrBarFill, { backgroundColor: colors.positive }]} />
            </View>
            {s.tp2 != null ? (
              <View style={[styles.rrBarSeg, { flex: s.tp2 - s.tp1, backgroundColor: colors.accentDim, borderTopRightRadius: 4, borderBottomRightRadius: 4 }]}>
                <View style={[styles.rrBarFill, { backgroundColor: colors.accent, opacity: 0.6 }]} />
              </View>
            ) : null}
          </View>
        </View>
      </Card>

      <Card>
        <SectionHeader label="Optional exits" hint="Layer in extra targets to let winners run further." />
        <ToggleRow
          label="Take Profit 2"
          desc="Scale out a second leg if price keeps moving"
          enabled={s.tp2 != null}
          onToggle={(on) => update({ tp2: on ? Math.max(s.tp1 + 1, s.tp1 * 1.6) : null })}
        />
        {s.tp2 != null ? (
          <Stepper
            label="TP2 level"
            value={s.tp2}
            onChange={(v) => update({ tp2: v })}
            min={s.tp1 + 0.5} max={100} step={0.5} unit="%" decimals={1}
            presets={[s.tp1 * 1.5, s.tp1 * 2, s.tp1 * 3].map(n => +n.toFixed(1))}
          />
        ) : null}

        <ToggleRow
          label="Trailing stop"
          desc="Move SL with price so profits lock in"
          enabled={s.trailing != null}
          onToggle={(on) => update({ trailing: on ? 1 : null })}
        />
        {s.trailing != null ? (
          <Stepper
            label="Trail distance"
            value={s.trailing}
            onChange={(v) => update({ trailing: v })}
            min={0.3} max={10} step={0.1} unit="%" decimals={1}
            presets={[0.5, 1, 1.5, 2, 3]}
          />
        ) : null}

        <ToggleRow
          label="Move SL to breakeven"
          desc="Once price moves favourably by X%, your stop becomes free"
          enabled={s.breakeven != null}
          onToggle={(on) => update({ breakeven: on ? 1 : null })}
        />
        {s.breakeven != null ? (
          <Stepper
            label="Breakeven trigger"
            value={s.breakeven}
            onChange={(v) => update({ breakeven: v })}
            min={0.3} max={20} step={0.1} unit="%" decimals={1}
            presets={[0.5, 1, 2, 3]}
          />
        ) : null}
      </Card>

      {warnings.length ? (
        <Card>
          <SectionHeader label="Heads-up" />
          {warnings.map((w, i) => (
            <View key={`w-${i}`} style={styles.warnRow}>
              <Pill
                label={w.tone === 'tip' ? '✓' : w.tone === 'warn' ? '!' : '⚠'}
                tone={w.tone === 'tip' ? 'positive' : w.tone === 'warn' ? 'warning' : 'negative'}
                small
              />
              <Text style={styles.warnTxt}>{w.msg}</Text>
            </View>
          ))}
        </Card>
      ) : null}

      <Card>
        <SectionHeader label="Strategy strength" icon="💪" />
        <StrengthMeter risk={risk} rr={Number(rr)} hasConfirms={false} hasTrailing={s.trailing != null} hasBE={s.breakeven != null} />
      </Card>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 6 — Risk + Universe + Filters
// ─────────────────────────────────────────────────────────────────────────
function Step6({ s, update }: { s: WizardState; update: (p: Partial<WizardState>) => void }) {
  const toggleSession = (id: Session) => {
    const has = s.sessions.includes(id);
    update({ sessions: has ? s.sessions.filter(x => x !== id) : [...s.sessions, id] });
  };
  const toggleDay = (id: Day) => {
    const has = s.tradingDays.includes(id);
    update({ tradingDays: has ? s.tradingDays.filter(x => x !== id) : [...s.tradingDays, id] });
  };
  return (
    <View>
      <StepIntro
        title="Risk & universe"
        subtitle="Position sizing, leverage, what coins to scan, and when the strategy is allowed to trade."
      />

      <Card>
        <SectionHeader label="Risk per trade" icon="⚙️" />
        <Stepper
          label="Leverage"
          value={s.leverage}
          onChange={(v) => update({ leverage: v })}
          min={1} max={125} step={1} unit="×"
          presets={[3, 5, 10, 15, 20, 50]}
        />
        <Stepper
          label="Position size"
          value={s.posSize}
          onChange={(v) => update({ posSize: v })}
          min={1} max={50} step={1} unit="%"
          presets={[2, 5, 8, 12, 20]}
          hint="% of account equity per trade"
        />
        <Stepper
          label="Max simultaneous positions"
          value={s.maxPos}
          onChange={(v) => update({ maxPos: v })}
          min={1} max={10} step={1}
          presets={[1, 2, 3, 5]}
        />
        <Stepper
          label="Max trades / day"
          value={s.maxTrades}
          onChange={(v) => update({ maxTrades: v })}
          min={1} max={50} step={1}
          presets={[2, 3, 5, 10, 20]}
        />
        <Stepper
          label="Cooldown between trades"
          value={s.cooldown}
          onChange={(v) => update({ cooldown: v })}
          min={0} max={1440} step={5} unit="m"
          presets={[5, 15, 30, 60, 240]}
        />
        <Stepper
          label="Daily loss limit"
          value={s.dailyLoss}
          onChange={(v) => update({ dailyLoss: v })}
          min={1} max={50} step={1} unit="%"
          presets={[3, 5, 8, 10, 15]}
          hint="Strategy auto-pauses if daily P&L hits this loss"
        />

        <SectionHeader label="Risk profile" />
        <ChipRow options={RISK_PROFILE_OPTIONS} value={s.riskProfile} onChange={(v) => update({ riskProfile: v })} size="sm" />

        <ToggleRow
          label="No duplicate coin per day"
          desc="Skip a coin if you already traded it today"
          enabled={s.noDuplicateSymbol}
          onToggle={(on) => update({ noDuplicateSymbol: on })}
        />
      </Card>

      <Card>
        <SectionHeader label="Coin universe" icon="🌐" />
        <ChipRow options={COIN_OPTIONS} value={s.coins} onChange={(v) => update({ coins: v })} size="sm" />

        {s.coins === 'single' ? (
          <View style={{ marginTop: spacing.sm }}>
            <Text style={styles.inputLabel}>Coin symbol</Text>
            <TextInput
              value={s.singleCoin}
              onChangeText={(t) => update({ singleCoin: t.toUpperCase() })}
              placeholder="e.g. BTC, SOL, ETH"
              placeholderTextColor={colors.textMute}
              autoCapitalize="characters"
              autoCorrect={false}
              style={styles.input}
            />
          </View>
        ) : null}
        {s.coins === 'specific' ? (
          <View style={{ marginTop: spacing.sm }}>
            <Text style={styles.inputLabel}>Watchlist (comma-separated)</Text>
            <TextInput
              value={s.specificCoins}
              onChangeText={(t) => update({ specificCoins: t })}
              placeholder="BTC, SOL, ETH, ADA"
              placeholderTextColor={colors.textMute}
              autoCapitalize="characters"
              autoCorrect={false}
              multiline
              style={[styles.input, { minHeight: 60 }]}
            />
          </View>
        ) : null}
        {(s.coins === 'gainers' || s.coins === 'losers') ? (
          <Stepper
            label={`24h ${s.coins === 'gainers' ? 'gain' : 'drop'} threshold`}
            value={s.coinsThreshold}
            onChange={(v) => update({ coinsThreshold: v })}
            min={1} max={50} step={1} unit="%"
            presets={[3, 5, 10, 15, 25]}
          />
        ) : null}
        {s.coins === 'all' ? (
          <Stepper
            label="Min 24h volume"
            value={s.minVol}
            onChange={(v) => update({ minVol: v })}
            min={500} max={50000} step={500} unit="k$"
            presets={[500, 1000, 5000, 10000]}
          />
        ) : null}
      </Card>

      <Card>
        <SectionHeader label="Sessions" icon="🕐" hint="Leave empty to trade 24/7" />
        <ChipRow
          options={SESSIONS.map(x => ({ value: x.id, label: x.label, hint: x.hint }))}
          value={s.sessions}
          onChange={toggleSession}
          multi
          size="sm"
        />
        <SectionHeader label="Trading days" icon="📅" hint="Leave empty for all 7 days" />
        <ChipRow
          options={DAYS.map(x => ({ value: x.id, label: x.label }))}
          value={s.tradingDays}
          onChange={toggleDay}
          multi
          size="sm"
        />
      </Card>

      <Card>
        <SectionHeader label="Market filters" icon="🛡️" />
        <ToggleRow
          label="HTF trend filter"
          desc="Only enter if 1H trend (EMA-20) aligns with trade direction"
          enabled={s.htfFilter}
          onToggle={(on) => update({ htfFilter: on })}
        />
        <View style={{ marginTop: spacing.sm }}>
          <Text style={styles.inputLabel}>BTC market regime</Text>
          <ChipRow
            options={BTC_REGIME_OPTIONS}
            value={s.btcRegime}
            onChange={(v) => update({ btcRegime: v })}
            size="sm"
          />
        </View>
      </Card>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Step 7 — Review / Launch
// ─────────────────────────────────────────────────────────────────────────
function Step7({
  s, update, fireRate, warnings, risk,
  nameSuggestions, nameLoading, onSuggestNames,
  btResult, btLoading, btError, onRunBacktest,
  saveResult, saving, onSave,
  publishLoading, publishDone, onPublish,
  onDone,
}: {
  s: WizardState;
  update: (p: Partial<WizardState>) => void;
  fireRate: string | null;
  warnings: ReturnType<typeof getWizardWarnings>;
  risk: ReturnType<typeof calcRiskLevel>;
  nameSuggestions: NameSuggestion[];
  nameLoading: boolean;
  onSuggestNames: () => void;
  btResult: BacktestResult | null;
  btLoading: boolean;
  btError: string | null;
  onRunBacktest: (days: 30 | 90) => void;
  saveResult: SaveResult | null;
  saving: boolean;
  onSave: () => void;
  publishLoading: boolean;
  publishDone: boolean;
  onPublish: () => void;
  onDone: () => void;
}) {
  return (
    <View>
      <StepIntro
        title="Review & launch"
        subtitle="Give your strategy a name, optionally backtest it, then save. You can publish to the marketplace afterwards."
      />

      <Card>
        <SectionHeader label="Strategy name" icon="🏷️" />
        <TextInput
          value={s.name}
          onChangeText={(t) => update({ name: t })}
          placeholder="e.g. Sniper FVG Long"
          placeholderTextColor={colors.textMute}
          maxLength={60}
          editable={!saveResult}
          style={styles.input}
        />
        <Pressable
          onPress={onSuggestNames}
          disabled={nameLoading || !!saveResult}
          style={({ pressed }) => [
            styles.aiBtn,
            (nameLoading || !!saveResult) && { opacity: 0.6 },
            pressed && { opacity: 0.85 },
          ]}
        >
          {nameLoading
            ? <ActivityIndicator color={colors.accent} size="small" />
            : <Text style={styles.aiIcon}>✨</Text>}
          <Text style={styles.aiBtnTxt}>
            {nameLoading ? 'Asking AI…' : '✨ Suggest names'}
          </Text>
        </Pressable>
        {nameSuggestions.length ? (
          <View style={{ marginTop: spacing.sm, gap: 6 }}>
            {nameSuggestions.map((n, i) => (
              <Pressable
                key={`ns-${i}`}
                onPress={() => update({ name: n.name })}
                style={({ pressed }) => [styles.suggestRow, pressed && { opacity: 0.85 }]}
              >
                <View style={{ flex: 1, minWidth: 0 }}>
                  <Text style={styles.suggestName} numberOfLines={1}>{n.name}</Text>
                  <Text style={styles.suggestTagline} numberOfLines={1}>{n.tagline}</Text>
                </View>
                <Ionicons name="arrow-up-circle-outline" size={18} color={colors.accent} />
              </Pressable>
            ))}
          </View>
        ) : null}
      </Card>

      <Card>
        <SectionHeader label="Strategy summary" />
        <SummaryGrid
          rows={[
            ['Style',     s.style ? `${STYLE_LABELS[s.style].icon} ${STYLE_LABELS[s.style].label}` : '—'],
            ['Direction', { LONG: '📈 Long', SHORT: '📉 Short', BOTH: '↕ Both' }[s.direction]],
            ['Mode',      s.mode === 'paper' ? '🧪 Paper' : '⚡ Live'],
            ['Timeframe', s.timeframe],
            ['Entry',     s.primaryType ? `${SIGNAL_META[s.primaryType].icon} ${SIGNAL_META[s.primaryType].label}` : '—'],
            ['Confirmations', s.confirms.length ? `${s.confirms.length} added` : 'None'],
            ['TP1 / SL',  `${s.tp1}% / ${s.sl}%${s.tp2 ? ` · TP2 ${s.tp2}%` : ''}${s.trailing ? ` · trail ${s.trailing}%` : ''}${s.breakeven ? ` · BE @${s.breakeven}%` : ''}`],
            ['Leverage',  `${s.leverage}× · ${risk.label}`],
            ['Position',  s.posSizeType === 'fixed' ? `$${s.posSizeUsd}` : `${s.posSize}% of equity`],
            ['Universe',
              s.coins === 'single'   ? `🎯 ${(s.singleCoin || 'BTC').replace(/USDT$/, '')}`
              : s.coins === 'gainers' ? `📈 Top gainers (${s.coinsThreshold}%+)`
              : s.coins === 'losers'  ? `📉 Top losers (${s.coinsThreshold}%+)`
              : s.coins === 'specific' ? `📋 Watchlist (${s.specificCoins.split(',').filter(Boolean).length} coins)`
              : '🌐 All coins'],
          ]}
        />
        {fireRate ? (
          <View style={{ marginTop: spacing.sm, flexDirection: 'row' }}>
            <Pill label={`🔥 Fires ${fireRate}`} tone="accent" small />
          </View>
        ) : null}
        {warnings.length ? (
          <View style={{ marginTop: spacing.sm }}>
            {warnings.map((w, i) => (
              <View key={`rw-${i}`} style={styles.warnRow}>
                <Pill
                  label={w.tone === 'tip' ? '✓' : w.tone === 'warn' ? '!' : '⚠'}
                  tone={w.tone === 'tip' ? 'positive' : w.tone === 'warn' ? 'warning' : 'negative'}
                  small
                />
                <Text style={styles.warnTxt}>{w.msg}</Text>
              </View>
            ))}
          </View>
        ) : null}
      </Card>

      <Card>
        <SectionHeader label="Quick backtest" icon="📊" hint="Replay your strategy on historical candles before saving." />
        <View style={{ flexDirection: 'row', gap: 8 }}>
          <Pressable
            onPress={() => onRunBacktest(30)}
            disabled={btLoading || !s.primaryType}
            style={({ pressed }) => [styles.btBtn, (btLoading || !s.primaryType) && { opacity: 0.55 }, pressed && { opacity: 0.85 }]}
          >
            {btLoading ? <ActivityIndicator color={colors.accent} size="small" /> : <Text style={styles.btBtnTxt}>30 days</Text>}
          </Pressable>
          <Pressable
            onPress={() => onRunBacktest(90)}
            disabled={btLoading || !s.primaryType}
            style={({ pressed }) => [styles.btBtn, (btLoading || !s.primaryType) && { opacity: 0.55 }, pressed && { opacity: 0.85 }]}
          >
            {btLoading ? <ActivityIndicator color={colors.accent} size="small" /> : <Text style={styles.btBtnTxt}>90 days</Text>}
          </Pressable>
        </View>
        {btError ? (
          <Text style={styles.btErr}>{btError}</Text>
        ) : null}
        {btResult?.stats ? (
          <BacktestResultCard r={btResult} />
        ) : null}
      </Card>

      {saveResult ? (
        <Card>
          <View style={styles.savedHeader}>
            <Text style={styles.savedIcon}>{s.mode === 'paper' ? '🧪' : '✅'}</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.savedTitle}>Strategy saved!</Text>
              <Text style={styles.savedSub}>"{saveResult.name}" is now in your library.</Text>
            </View>
          </View>

          <SectionHeader
            label="Publish to marketplace"
            icon="🌟"
            hint="Share your strategy with the TradeHub community — anyone can copy it for free. Paid &amp; subscription pricing can be configured later from the web app."
          />

          {publishDone ? (
            <View style={[styles.savedHeader, { marginTop: spacing.sm }]}>
              <Text style={styles.savedIcon}>🎉</Text>
              <Text style={styles.savedSub}>Your strategy is live on the marketplace!</Text>
            </View>
          ) : (
            <View style={{ marginTop: spacing.sm }}>
              <PrimaryButton
                label={publishLoading ? 'Publishing…' : '🌟 Publish to Marketplace'}
                onPress={onPublish}
                loading={publishLoading}
              />
            </View>
          )}

          <View style={{ marginTop: spacing.sm }}>
            <PrimaryButton label="Done — view my strategies" onPress={onDone} variant="secondary" />
          </View>
        </Card>
      ) : (
        <Card>
          <PrimaryButton
            label={saving ? 'Saving…' : '✅ Save Strategy'}
            onPress={onSave}
            loading={saving}
            disabled={!s.name.trim() || !s.primaryType}
          />
          <Text style={styles.hint}>
            Saving creates the strategy in {s.mode === 'paper' ? 'paper-trade' : 'live'} mode. You can edit or delete it later.
          </Text>
        </Card>
      )}
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Small inline subcomponents
// ─────────────────────────────────────────────────────────────────────────
function StepIntro({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <View style={{ marginBottom: spacing.md }}>
      <Text style={styles.stepTitle}>{title}</Text>
      <Text style={styles.stepSub}>{subtitle}</Text>
    </View>
  );
}

function Card({ children }: { children: React.ReactNode }) {
  return <View style={styles.card}>{children}</View>;
}

function ToggleRow({
  label, desc, enabled, onToggle,
}: {
  label: string; desc?: string; enabled: boolean; onToggle: (next: boolean) => void;
}) {
  return (
    <Pressable
      onPress={() => {
        if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
        onToggle(!enabled);
      }}
      style={({ pressed }) => [styles.toggleRow, pressed && { opacity: 0.85 }]}
      accessibilityRole="switch"
      accessibilityState={{ checked: enabled }}
    >
      <View style={{ flex: 1, paddingRight: 10, minWidth: 0 }}>
        <Text style={styles.toggleLabel}>{label}</Text>
        {desc ? <Text style={styles.toggleDesc}>{desc}</Text> : null}
      </View>
      <View style={[styles.toggleTrack, enabled && styles.toggleTrackOn]}>
        <View style={[styles.toggleKnob, enabled && styles.toggleKnobOn]} />
      </View>
    </Pressable>
  );
}

function SummaryGrid({ rows }: { rows: [string, string][] }) {
  return (
    <View style={styles.summaryGrid}>
      {rows.map(([k, v]) => (
        <View key={k} style={styles.summaryRow}>
          <Text style={styles.summaryKey}>{k}</Text>
          <Text style={styles.summaryVal} numberOfLines={1}>{v}</Text>
        </View>
      ))}
    </View>
  );
}

function BacktestResultCard({ r }: { r: BacktestResult }) {
  const st = r.stats!;
  const trades = st.closed_trades ?? st.total_trades ?? 0;
  const wr = st.win_rate ?? 0;
  const pnl = st.total_pnl ?? 0;
  const pf = st.profit_factor ?? 0;
  const dd = st.max_drawdown ?? 0;
  const wrTone = wr >= 60 ? 'positive' : wr >= 45 ? 'warning' : 'negative';
  const pnlTone = pnl >= 0 ? 'positive' : 'negative';
  const pfTone = pf >= 1.5 ? 'positive' : pf >= 1 ? 'warning' : 'negative';
  return (
    <View style={{ marginTop: spacing.sm, gap: 6 }}>
      <View style={styles.btSymRow}>
        <Text style={styles.btSym}>📊 {r.symbol || 'BTCUSDT'}</Text>
        <Text style={styles.btSymSub}>{r.timeframe || '5m'} · {r.days ?? 30}d · {trades} trade{trades === 1 ? '' : 's'}</Text>
      </View>
      <View style={styles.btStatGrid}>
        <BtStat label="Win rate"     value={`${wr.toFixed(1)}%`}      tone={wrTone} />
        <BtStat label="Total P&L"    value={`${pnl >= 0 ? '+' : ''}${pnl.toFixed(1)}%`} tone={pnlTone} />
        <BtStat label="Profit factor" value={pf ? pf.toFixed(2) : '—'} tone={pfTone} />
        <BtStat label="Max drawdown" value={`-${dd.toFixed(1)}%`}     tone="warning" />
      </View>
      {trades === 0 ? (
        <Text style={styles.btErr}>No trades triggered in this period — try relaxing your conditions or a different timeframe.</Text>
      ) : null}
    </View>
  );
}

function BtStat({ label, value, tone }: { label: string; value: string; tone: 'positive'|'negative'|'warning'|'accent'|'neutral' }) {
  const colorMap = {
    positive: colors.positive, negative: colors.negative,
    warning: colors.warning, accent: colors.accent, neutral: colors.text,
  } as const;
  return (
    <View style={styles.btStatBox}>
      <Text style={[styles.btStatVal, { color: colorMap[tone] }]}>{value}</Text>
      <Text style={styles.btStatLabel}>{label}</Text>
    </View>
  );
}

function StrengthMeter({
  risk, rr, hasConfirms, hasTrailing, hasBE,
}: {
  risk: ReturnType<typeof calcRiskLevel>;
  rr: number;
  hasConfirms: boolean;
  hasTrailing: boolean;
  hasBE: boolean;
}) {
  const checks = [
    { label: 'R:R ratio 2:1+', ok: rr >= 2 },
    { label: 'Confirmation signals', ok: hasConfirms },
    { label: 'Trailing stop', ok: hasTrailing },
    { label: 'Breakeven protection', ok: hasBE },
    { label: 'Conservative risk', ok: risk.label === 'CONSERVATIVE' || risk.label === 'MODERATE' },
  ];
  const score = checks.filter(c => c.ok).length;
  const pct = (score / checks.length) * 100;
  const barColor = pct >= 80 ? colors.positive : pct >= 60 ? colors.accent : pct >= 40 ? colors.warning : colors.negative;
  const grade = pct >= 80 ? 'A' : pct >= 60 ? 'B' : pct >= 40 ? 'C' : 'D';

  return (
    <View>
      <View style={strengthStyles.header}>
        <Text style={[strengthStyles.grade, { color: barColor }]}>{grade}</Text>
        <View style={{ flex: 1 }}>
          <View style={strengthStyles.barTrack}>
            <View style={[strengthStyles.barFill, { width: `${pct}%`, backgroundColor: barColor }]} />
          </View>
          <Text style={strengthStyles.pctTxt}>{score}/{checks.length} checks passed</Text>
        </View>
      </View>
      <View style={strengthStyles.list}>
        {checks.map((c, i) => (
          <View key={`sc-${i}`} style={strengthStyles.row}>
            <Text style={{ fontSize: 12 }}>{c.ok ? '✅' : '⬜'}</Text>
            <Text style={[strengthStyles.rowTxt, c.ok && { color: colors.text }]}>{c.label}</Text>
          </View>
        ))}
      </View>
    </View>
  );
}

const strengthStyles = StyleSheet.create({
  header: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 8 },
  grade: { fontFamily: font.bold, fontSize: 28, width: 36, textAlign: 'center' },
  barTrack: { height: 6, borderRadius: 3, backgroundColor: colors.bgElev, overflow: 'hidden' },
  barFill: { height: '100%', borderRadius: 3 },
  pctTxt: { fontFamily: font.medium, fontSize: 11, color: colors.textMute, marginTop: 4 },
  list: { gap: 4 },
  row: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  rowTxt: { fontFamily: font.regular, fontSize: 12.5, color: colors.textMute },
});

// ─────────────────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  root:    { flex: 1, backgroundColor: colors.bg },

  progressWrap: {
    paddingHorizontal: spacing.lg, paddingTop: spacing.sm,
  },
  progressTrack: {
    height: 3, borderRadius: 1.5, backgroundColor: colors.bgElev,
    marginBottom: spacing.sm, overflow: 'hidden',
  },
  progressFill: {
    height: '100%', borderRadius: 1.5, backgroundColor: colors.accent,
  },

  dotsRow: {
    flexDirection: 'row', justifyContent: 'space-between',
    paddingHorizontal: 0, paddingTop: 0, paddingBottom: 4,
  },
  dotWrap: { flex: 1, alignItems: 'center' },
  dot: {
    width: 32, height: 32, borderRadius: 16,
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border,
    alignItems: 'center', justifyContent: 'center', marginBottom: 3,
  },
  dotActive: { borderColor: colors.accent, backgroundColor: colors.accentDim },
  dotDone:   { borderColor: colors.positive, backgroundColor: 'rgba(52,211,153,0.18)' },
  dotIcon:   { fontSize: 14, color: colors.textDim },
  dotIconActive: { color: colors.text },
  dotLabel:  { fontFamily: font.medium, fontSize: 9.5, color: colors.textMute },
  dotLabelActive: { color: colors.accent, fontFamily: font.semibold },

  summaryBar: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    paddingHorizontal: spacing.lg, paddingBottom: spacing.sm,
  },
  summaryStep: {
    fontFamily: font.bold, fontSize: 11.5, color: colors.accent,
  },
  summary: {
    fontFamily: font.medium, fontSize: 11.5, color: colors.textDim,
    flex: 1,
  },

  body:    { paddingHorizontal: spacing.lg, paddingTop: spacing.sm },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.lg, borderWidth: 1, borderColor: colors.border,
    padding: spacing.md, marginBottom: spacing.md,
  },

  stepTitle: { fontFamily: font.bold, fontSize: 20, color: colors.text, marginBottom: 4 },
  stepSub:   { fontFamily: font.regular, fontSize: 13.5, color: colors.textDim, lineHeight: 19 },
  hint:      { fontFamily: font.regular, fontSize: 12, color: colors.textMute, marginTop: 6, lineHeight: 16 },

  styleGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: spacing.md },

  previewCard: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.borderHi,
    padding: 12, marginTop: 4,
  },
  previewTitle: { fontFamily: font.semibold, fontSize: 13, color: colors.text, marginBottom: 4 },
  previewLine:  { fontFamily: font.regular, fontSize: 12, color: colors.textDim, lineHeight: 17 },
  previewVal:   { fontFamily: font.bold, color: colors.text },

  signalPickRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.borderHi,
    paddingVertical: 12, paddingHorizontal: 12,
  },
  signalIcon:  { fontSize: 22, width: 28, textAlign: 'center' },
  signalLabel: { fontFamily: font.semibold, fontSize: 14, color: colors.text },
  signalDesc:  { fontFamily: font.regular, fontSize: 12, color: colors.textMute, marginTop: 2 },

  fireRow: { flexDirection: 'row', marginTop: spacing.sm },

  confirmsHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  addBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingHorizontal: 10, paddingVertical: 6,
    borderRadius: radius.pill,
    backgroundColor: colors.accentDim, borderWidth: 1, borderColor: 'rgba(255,255,255,0.10)',
  },
  addBtnTxt: { fontFamily: font.semibold, fontSize: 12, color: colors.accent },
  emptyBox: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.border,
    padding: 12, marginTop: 6,
  },
  emptyTitle: { fontFamily: font.semibold, fontSize: 13, color: colors.text },
  emptyHint:  { fontFamily: font.regular, fontSize: 12, color: colors.textMute, marginTop: 4, lineHeight: 16 },
  confirmRow: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.border,
    marginTop: 8, overflow: 'hidden',
  },
  confirmHead: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    padding: 12,
  },
  confirmIcon:  { fontSize: 18, width: 24, textAlign: 'center' },
  confirmLabel: { fontFamily: font.semibold, fontSize: 13.5, color: colors.text },
  confirmDesc:  { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute, marginTop: 2 },
  removeBtn: {
    width: 24, height: 24, borderRadius: 12,
    backgroundColor: colors.negativeDim,
    alignItems: 'center', justifyContent: 'center',
  },
  confirmBody: {
    paddingHorizontal: 12, paddingBottom: 12,
    borderTopWidth: 1, borderTopColor: colors.border,
  },

  rrRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 4 },
  rrHint:{ fontFamily: font.regular, fontSize: 12, color: colors.textDim },

  rrBarWrap: { marginTop: spacing.md },
  rrBarLabel: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 4 },
  rrBarTxt: { fontFamily: font.medium, fontSize: 10.5 },
  rrBarTrack: { flexDirection: 'row', height: 14, borderRadius: 4, overflow: 'hidden', backgroundColor: colors.bgElev },
  rrBarSeg: { justifyContent: 'center', overflow: 'hidden' },
  rrBarFill: { height: '100%', opacity: 0.4 },
  rrBarEntry: { width: 2, height: '100%' },

  toggleRow: {
    flexDirection: 'row', alignItems: 'center',
    paddingVertical: 10,
    borderTopWidth: 1, borderTopColor: colors.border,
  },
  toggleLabel: { fontFamily: font.semibold, fontSize: 13.5, color: colors.text },
  toggleDesc:  { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute, marginTop: 2, lineHeight: 15 },
  toggleTrack: {
    width: 42, height: 24, borderRadius: 12,
    backgroundColor: colors.bgElev, borderWidth: 1, borderColor: colors.border,
    padding: 2, justifyContent: 'center',
  },
  toggleTrackOn: { backgroundColor: colors.accentDim, borderColor: colors.accent },
  toggleKnob:    { width: 18, height: 18, borderRadius: 9, backgroundColor: colors.textMute },
  toggleKnobOn:  { backgroundColor: colors.accent, transform: [{ translateX: 18 }] },

  warnRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 6 },
  warnTxt: { flex: 1, fontFamily: font.regular, fontSize: 12, color: colors.textDim, lineHeight: 16 },

  riskBadgeRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', paddingHorizontal: 4, marginTop: 4 },
  riskBadgeLabel: { fontFamily: font.medium, fontSize: 12, color: colors.textDim },

  inputLabel: { fontFamily: font.medium, fontSize: 11.5, color: colors.textDim, textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 6 },
  input: {
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.borderHi,
    paddingHorizontal: 12, paddingVertical: Platform.OS === 'ios' ? 12 : 8,
    fontFamily: font.medium, fontSize: 14, color: colors.text,
  },

  aiBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8,
    marginTop: spacing.sm, paddingVertical: 10,
    borderRadius: radius.md,
    backgroundColor: colors.violetDim, borderWidth: 1, borderColor: 'rgba(255,255,255,0.10)',
  },
  aiIcon:    { fontSize: 14 },
  aiBtnTxt:  { fontFamily: font.semibold, fontSize: 13, color: colors.violet },
  suggestRow: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    paddingVertical: 10, paddingHorizontal: 12,
    borderRadius: radius.md, backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.border,
  },
  suggestName:    { fontFamily: font.semibold, fontSize: 13.5, color: colors.text },
  suggestTagline: { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute, marginTop: 2 },

  summaryGrid: { backgroundColor: colors.bgElev, borderRadius: radius.md, padding: 12 },
  summaryRow:  { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  summaryKey:  { fontFamily: font.regular, fontSize: 12, color: colors.textMute, flexShrink: 0 },
  summaryVal:  { fontFamily: font.semibold, fontSize: 12.5, color: colors.text, flex: 1, textAlign: 'right', paddingLeft: 12 },

  btBtn: {
    flex: 1, paddingVertical: 11,
    borderRadius: radius.md,
    backgroundColor: colors.bgElev, borderWidth: 1, borderColor: colors.borderHi,
    alignItems: 'center',
  },
  btBtnTxt:  { fontFamily: font.semibold, fontSize: 13, color: colors.text },
  btErr:     { fontFamily: font.regular, fontSize: 12, color: colors.negative, marginTop: 8, lineHeight: 16 },
  btSymRow:  { flexDirection: 'row', alignItems: 'baseline', gap: 8 },
  btSym:     { fontFamily: font.semibold, fontSize: 13, color: colors.text },
  btSymSub:  { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute },
  btStatGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  btStatBox: {
    flexBasis: '48%', flexGrow: 1,
    backgroundColor: colors.bgElev,
    borderRadius: radius.md, borderWidth: 1, borderColor: colors.border,
    padding: 10, alignItems: 'center',
  },
  btStatVal:   { fontFamily: font.bold, fontSize: 16 },
  btStatLabel: { fontFamily: font.regular, fontSize: 10.5, color: colors.textMute, marginTop: 2 },

  savedHeader: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: spacing.md },
  savedIcon:   { fontSize: 22 },
  savedTitle:  { fontFamily: font.bold, fontSize: 15, color: colors.text },
  savedSub:    { fontFamily: font.regular, fontSize: 12.5, color: colors.textDim, marginTop: 2 },

  errorBox: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    marginTop: spacing.md,
    padding: 12, borderRadius: radius.md,
    backgroundColor: colors.negativeDim,
    borderWidth: 1, borderColor: 'rgba(248,113,113,0.36)',
  },
  errorTxt: { flex: 1, fontFamily: font.medium, fontSize: 12.5, color: colors.negative, lineHeight: 16 },

  footer: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingHorizontal: spacing.lg, paddingTop: spacing.sm,
    backgroundColor: colors.bg, borderTopWidth: 1, borderTopColor: colors.border,
    gap: 12,
  },
  backBtn: {
    flexDirection: 'row', alignItems: 'center', gap: 4,
    paddingVertical: 12, paddingHorizontal: 14,
    borderRadius: radius.md, backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
  },
  backTxt: { fontFamily: font.semibold, fontSize: 13, color: colors.text },
});
