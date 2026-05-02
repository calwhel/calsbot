import React, { useCallback, useMemo, useState } from 'react';
import {
  View, Text, StyleSheet, ScrollView, TextInput,
  Pressable, Alert, KeyboardAvoidingView, Platform, ActivityIndicator,
} from 'react-native';
import { Stack, useRouter } from 'expo-router';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { PrimaryButton } from '@/components/PrimaryButton';
import { Pill } from '@/components/Pill';
import { colors, font, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiPost, type SaveStrategyResponse } from '@/lib/api';

type Direction = 'LONG' | 'SHORT';
type UniverseMode = 'all' | 'specific';
type Tf = '5m' | '15m' | '1h' | '4h';
type Operator = 'lt' | 'gt';

type WizardState = {
  name: string;
  direction: Direction;
  universe: UniverseMode;
  symbols: string;             // comma-separated; only used when universe==='specific'
  rsiOperator: Operator;
  rsiValue: string;            // user-entered, parsed before submit
  rsiTimeframe: Tf;
  takeProfitPct: string;
  stopLossPct: string;
  leverage: number;
};

const STEP_COUNT = 3;

function StepDots({ step }: { step: number }) {
  return (
    <View style={styles.dots}>
      {Array.from({ length: STEP_COUNT }).map((_, i) => (
        <View
          key={`d-${i}`}
          style={[
            styles.dot,
            i === step && styles.dotActive,
            i < step && styles.dotDone,
          ]}
        />
      ))}
    </View>
  );
}

function ChoiceRow<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { v: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <View style={styles.choices}>
      {options.map((o) => {
        const active = value === o.v;
        return (
          <Pressable
            key={o.v}
            onPress={() => onChange(o.v)}
            style={[styles.choice, active && styles.choiceActive]}
          >
            <Text style={[styles.choiceText, active && styles.choiceTextActive]}>
              {o.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );
}

function FieldLabel({ children, hint }: { children: React.ReactNode; hint?: string }) {
  return (
    <View style={{ marginBottom: spacing.sm }}>
      <Text style={styles.label}>{children}</Text>
      {hint ? <Text style={styles.hint}>{hint}</Text> : null}
    </View>
  );
}

export default function WizardScreen() {
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();
  const qc = useQueryClient();
  const [step, setStep] = useState(0);

  const [s, setS] = useState<WizardState>({
    name: '',
    direction: 'LONG',
    universe: 'all',
    symbols: '',
    rsiOperator: 'lt',
    rsiValue: '30',
    rsiTimeframe: '15m',
    takeProfitPct: '3',
    stopLossPct: '1.5',
    leverage: 10,
  });

  const update = useCallback(<K extends keyof WizardState>(key: K, val: WizardState[K]) => {
    setS((p) => ({ ...p, [key]: val }));
  }, []);

  // ─── Validation per step ──────────────────────────────────────────────────
  const stepError = useMemo<string | null>(() => {
    if (step === 0) {
      if (!s.name.trim()) return 'Strategy name is required.';
      if (s.name.trim().length < 3) return 'Strategy name must be at least 3 characters.';
      return null;
    }
    if (step === 1) {
      if (s.universe === 'specific') {
        const syms = s.symbols.split(',').map((x) => x.trim().toUpperCase()).filter(Boolean);
        if (syms.length === 0) return 'List at least one coin (e.g. BTCUSDT, ETHUSDT).';
        for (const sym of syms) {
          if (!/^[A-Z0-9]{4,15}$/.test(sym)) return `"${sym}" doesn’t look like a valid symbol.`;
        }
      }
      const rsi = Number(s.rsiValue);
      if (!Number.isFinite(rsi) || rsi < 1 || rsi > 99) return 'RSI threshold must be between 1 and 99.';
      return null;
    }
    if (step === 2) {
      const tp = Number(s.takeProfitPct);
      const sl = Number(s.stopLossPct);
      if (!Number.isFinite(tp) || tp <= 0 || tp > 50) return 'Take profit must be between 0.1% and 50%.';
      if (!Number.isFinite(sl) || sl <= 0 || sl > 50) return 'Stop loss must be between 0.1% and 50%.';
      if (s.leverage < 1 || s.leverage > 100) return 'Leverage must be between 1× and 100×.';
      return null;
    }
    return null;
  }, [s, step]);

  // ─── Build the strategy config from wizard state ──────────────────────────
  const buildConfig = useCallback(() => {
    const symbols = s.symbols.split(',').map((x) => x.trim().toUpperCase()).filter(Boolean);
    const tp = Number(s.takeProfitPct);
    const sl = Number(s.stopLossPct);
    const rsiVal = Number(s.rsiValue);
    return {
      version: '1.0',
      name: s.name.trim(),
      description: `Mobile-built · ${s.direction} when RSI ${s.rsiOperator === 'lt' ? '<' : '>'} ${rsiVal} on ${s.rsiTimeframe}`,
      direction: s.direction,
      universe: s.universe === 'all'
        ? { type: 'all' }
        : { type: 'specific', symbols },
      entry_conditions: {
        operator: 'AND',
        conditions: [
          {
            type: 'indicator',
            name: 'rsi',
            timeframe: s.rsiTimeframe,
            operator: s.rsiOperator,
            value: rsiVal,
          },
        ],
      },
      exit: {
        take_profit_pct: tp,
        take_profit2_pct: null,
        stop_loss_pct: sl,
        trailing_stop: false,
        trailing_stop_pct: null,
        breakeven_at_pct: null,
      },
      risk: {
        leverage: s.leverage,
        position_size_pct: 5,
        max_trades_per_day: 5,
        max_open_positions: 3,
        cooldown_minutes: 30,
        daily_loss_limit_pct: 10,
        no_duplicate_symbol: true,
        risk_profile: 'medium',
        position_size_type: 'pct',
      },
      filters: {},
      _build_mode: 'paper',
      _category: 'custom',
      _timeframe: s.rsiTimeframe,
      _source: 'mobile_wizard',
    };
  }, [s]);

  const saveM = useMutation({
    mutationFn: () => apiPost<SaveStrategyResponse>('/api/save-strategy', {
      uid,
      config: buildConfig(),
    }),
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ['strategies', uid] });
      qc.invalidateQueries({ queryKey: ['portfolio', uid] });
      Alert.alert(
        'Strategy created',
        `“${resp.name}” is in paper-trading mode. Open it to review or activate.`,
        [
          {
            text: 'View it',
            onPress: () => router.replace(`/strategy/${resp.id}` as any),
          },
        ],
      );
    },
    onError: (e) => {
      Alert.alert('Could not create strategy', (e as Error).message || 'Try again.');
    },
  });

  const next = useCallback(() => {
    if (stepError) {
      Alert.alert('Fix this first', stepError);
      return;
    }
    if (step < STEP_COUNT - 1) setStep(step + 1);
    else saveM.mutate();
  }, [step, stepError, saveM]);

  const back = useCallback(() => {
    if (step > 0) setStep(step - 1);
    else router.back();
  }, [step, router]);

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <>
      <Stack.Screen options={{ title: 'New strategy', headerShown: true, headerStyle: { backgroundColor: colors.bg }, headerTintColor: colors.text, headerShadowVisible: false }} />
      <KeyboardAvoidingView
        style={{ flex: 1, backgroundColor: colors.bg }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={insets.top + 60}
      >
        <ScrollView
          contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 120 }]}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <StepDots step={step} />

          {step === 0 && (
            <View>
              <Text style={styles.title}>The basics</Text>
              <Text style={styles.subtitle}>What should we call it, and which side are you taking?</Text>

              <View style={{ marginTop: spacing.lg }}>
                <FieldLabel>Strategy name</FieldLabel>
                <TextInput
                  style={styles.input}
                  value={s.name}
                  onChangeText={(v) => update('name', v)}
                  placeholder="e.g. BTC oversold bounce"
                  placeholderTextColor={colors.textMute}
                  autoCapitalize="sentences"
                  maxLength={50}
                />
              </View>

              <View style={{ marginTop: spacing.lg }}>
                <FieldLabel hint="LONG buys the dip; SHORT bets against the move.">Direction</FieldLabel>
                <ChoiceRow
                  value={s.direction}
                  onChange={(v) => update('direction', v)}
                  options={[
                    { v: 'LONG', label: 'LONG' },
                    { v: 'SHORT', label: 'SHORT' },
                  ]}
                />
              </View>
            </View>
          )}

          {step === 1 && (
            <View>
              <Text style={styles.title}>Coins & signal</Text>
              <Text style={styles.subtitle}>Which coins to scan, and the entry trigger.</Text>

              <View style={{ marginTop: spacing.lg }}>
                <FieldLabel hint="Scan every coin or restrict to a specific list.">Universe</FieldLabel>
                <ChoiceRow
                  value={s.universe}
                  onChange={(v) => update('universe', v)}
                  options={[
                    { v: 'all', label: 'All coins' },
                    { v: 'specific', label: 'Specific list' },
                  ]}
                />
                {s.universe === 'specific' && (
                  <TextInput
                    style={[styles.input, { marginTop: spacing.sm }]}
                    value={s.symbols}
                    onChangeText={(v) => update('symbols', v)}
                    placeholder="BTCUSDT, ETHUSDT, SOLUSDT"
                    placeholderTextColor={colors.textMute}
                    autoCapitalize="characters"
                    autoCorrect={false}
                  />
                )}
              </View>

              <View style={{ marginTop: spacing.lg }}>
                <FieldLabel hint="Trigger when RSI crosses a threshold on the chosen timeframe.">Entry condition · RSI</FieldLabel>
                <View style={{ flexDirection: 'row', gap: spacing.sm }}>
                  <View style={{ flex: 1 }}>
                    <ChoiceRow
                      value={s.rsiOperator}
                      onChange={(v) => update('rsiOperator', v)}
                      options={[
                        { v: 'lt', label: 'RSI <' },
                        { v: 'gt', label: 'RSI >' },
                      ]}
                    />
                  </View>
                  <TextInput
                    style={[styles.input, { width: 80, textAlign: 'center' }]}
                    value={s.rsiValue}
                    onChangeText={(v) => update('rsiValue', v.replace(/[^0-9]/g, ''))}
                    keyboardType="number-pad"
                    maxLength={2}
                  />
                </View>
                <View style={{ marginTop: spacing.sm }}>
                  <ChoiceRow
                    value={s.rsiTimeframe}
                    onChange={(v) => update('rsiTimeframe', v)}
                    options={[
                      { v: '5m',  label: '5m'  },
                      { v: '15m', label: '15m' },
                      { v: '1h',  label: '1h'  },
                      { v: '4h',  label: '4h'  },
                    ]}
                  />
                </View>
              </View>
            </View>
          )}

          {step === 2 && (
            <View>
              <Text style={styles.title}>Risk profile</Text>
              <Text style={styles.subtitle}>Where do you take profit, and how much can you lose?</Text>

              <View style={{ flexDirection: 'row', marginTop: spacing.lg, gap: spacing.md }}>
                <View style={{ flex: 1 }}>
                  <FieldLabel>Take profit (%)</FieldLabel>
                  <TextInput
                    style={styles.input}
                    value={s.takeProfitPct}
                    onChangeText={(v) => update('takeProfitPct', v.replace(/[^0-9.]/g, ''))}
                    keyboardType="decimal-pad"
                    maxLength={5}
                  />
                </View>
                <View style={{ flex: 1 }}>
                  <FieldLabel>Stop loss (%)</FieldLabel>
                  <TextInput
                    style={styles.input}
                    value={s.stopLossPct}
                    onChangeText={(v) => update('stopLossPct', v.replace(/[^0-9.]/g, ''))}
                    keyboardType="decimal-pad"
                    maxLength={5}
                  />
                </View>
              </View>

              <View style={{ marginTop: spacing.lg }}>
                <FieldLabel hint="Bitunix futures leverage. Higher = bigger gains and bigger liquidation risk.">Leverage</FieldLabel>
                <View style={styles.choices}>
                  {[5, 10, 20, 50, 75].map((lev) => {
                    const active = s.leverage === lev;
                    return (
                      <Pressable
                        key={`lev-${lev}`}
                        onPress={() => update('leverage', lev)}
                        style={[styles.choice, active && styles.choiceActive]}
                      >
                        <Text style={[styles.choiceText, active && styles.choiceTextActive]}>
                          {lev}×
                        </Text>
                      </Pressable>
                    );
                  })}
                </View>
              </View>

              <View style={{ marginTop: spacing.lg }}>
                <Text style={styles.summaryLabel}>You’re creating</Text>
                <View style={styles.summaryCard}>
                  <Text style={styles.summaryName}>{s.name || '(unnamed)'}</Text>
                  <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: spacing.sm }}>
                    <Pill label={s.direction} tone={s.direction === 'LONG' ? 'positive' : 'negative'} small />
                    <Pill label={s.universe === 'all' ? 'All coins' : `${s.symbols.split(',').filter((x) => x.trim()).length} coin(s)`} tone="neutral" small />
                    <Pill label={`RSI ${s.rsiOperator === 'lt' ? '<' : '>'} ${s.rsiValue} · ${s.rsiTimeframe}`} tone="accent" small />
                    <Pill label={`TP ${s.takeProfitPct}% · SL ${s.stopLossPct}%`} tone="warning" small />
                    <Pill label={`${s.leverage}× lev`} tone="neutral" small />
                  </View>
                  <Text style={styles.summaryNote}>
                    Starts in paper-trading mode. You can flip it live from the strategy
                    detail screen once you’re confident.
                  </Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>

        <View style={[styles.footer, { paddingBottom: insets.bottom + 12 }]}>
          <Pressable onPress={back} style={styles.backBtn} disabled={saveM.isPending}>
            <Ionicons name="chevron-back" size={20} color={colors.textDim} />
            <Text style={styles.backBtnText}>{step === 0 ? 'Cancel' : 'Back'}</Text>
          </Pressable>
          <View style={{ flex: 1 }}>
            <PrimaryButton
              label={step === STEP_COUNT - 1 ? 'Create strategy' : 'Next'}
              onPress={next}
              loading={saveM.isPending}
            />
          </View>
        </View>
      </KeyboardAvoidingView>
    </>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.md },
  dots: { flexDirection: 'row', gap: 6, marginBottom: spacing.lg },
  dot: { width: 30, height: 4, borderRadius: 2, backgroundColor: colors.border },
  dotActive: { backgroundColor: colors.accent },
  dotDone: { backgroundColor: colors.positive },
  title: { color: colors.text, fontFamily: font.black, fontSize: 26, letterSpacing: -0.6 },
  subtitle: { color: colors.textDim, fontFamily: font.regular, fontSize: 14, marginTop: 6, lineHeight: 20 },
  label: { color: colors.text, fontFamily: font.bold, fontSize: 12, letterSpacing: 0.6, textTransform: 'uppercase' },
  hint: { color: colors.textMute, fontFamily: font.regular, fontSize: 12, marginTop: 3, lineHeight: 16 },
  input: {
    backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md, paddingVertical: 12,
    color: colors.text, fontFamily: font.medium, fontSize: 15,
  },
  choices: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  choice: {
    paddingHorizontal: 14, paddingVertical: 10,
    borderRadius: radius.md,
    borderWidth: 1, borderColor: colors.border,
    backgroundColor: colors.card,
  },
  choiceActive: {
    borderColor: colors.accent,
    backgroundColor: colors.accentDim,
  },
  choiceText: { color: colors.textDim, fontFamily: font.bold, fontSize: 13, letterSpacing: 0.3 },
  choiceTextActive: { color: colors.accent },
  summaryLabel: {
    color: colors.textDim, fontFamily: font.bold, fontSize: 12,
    letterSpacing: 0.6, textTransform: 'uppercase', marginBottom: spacing.sm,
  },
  summaryCard: {
    backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
    borderRadius: radius.lg, padding: spacing.lg,
  },
  summaryName: { color: colors.text, fontFamily: font.black, fontSize: 19, letterSpacing: -0.3 },
  summaryNote: { color: colors.textMute, fontFamily: font.regular, fontSize: 12, marginTop: spacing.md, lineHeight: 17 },
  footer: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: spacing.lg, paddingTop: spacing.md,
    backgroundColor: colors.bg,
    borderTopWidth: 1, borderTopColor: colors.border,
    gap: spacing.md,
  },
  backBtn: { flexDirection: 'row', alignItems: 'center', paddingHorizontal: 4, paddingVertical: 12 },
  backBtnText: { color: colors.textDim, fontFamily: font.semibold, fontSize: 14, marginLeft: 2 },
});
