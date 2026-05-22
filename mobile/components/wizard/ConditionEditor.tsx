import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ChipRow, type ChipOption } from './ChipRow';
import { Stepper } from './Stepper';
import { type SignalType, type Tf, SIGNAL_META } from '@/lib/strategyPresets';
import { colors, font, radius, spacing } from '@/constants/colors';

const TF_OPTIONS: ChipOption<Tf>[] = [
  { value: '1m', label: '1m' }, { value: '5m', label: '5m' }, { value: '15m', label: '15m' },
  { value: '1h', label: '1h' }, { value: '4h', label: '4h' }, { value: '1d', label: '1d' },
];

/**
 * Inline editor for a single signal config. Used for both the primary entry
 * signal and for each confirmation. Renders the per-type knobs (condition,
 * level, multiplier, pattern, etc.) with the same defaults as the web
 * `renderSignalCfg(type, cfg)` switch in app/templates/strategy_portal.html.
 */
export function ConditionEditor({
  type,
  cfg,
  onChange,
  showTimeframe = true,
  compact = false,
}: {
  type: SignalType;
  cfg: Record<string, any>;
  onChange: (next: Record<string, any>) => void;
  showTimeframe?: boolean;
  /** Confirmation rows render in a slimmer card; primary uses full padding. */
  compact?: boolean;
}) {
  const set = (patch: Record<string, any>) => onChange({ ...cfg, ...patch });
  const tf: Tf = (cfg.timeframe as Tf) || '15m';
  const meta = SIGNAL_META[type];

  // ── Timeframe (most signals support it; price_momentum & vwap_deviation
  //    are inherently per-window so timeframe doesn't apply there) ─────────
  const TF = showTimeframe && type !== 'price_momentum' && type !== 'vwap_deviation' ? (
    <Section label="Timeframe" compact={compact}>
      <ChipRow options={TF_OPTIONS} value={tf} onChange={(v) => set({ timeframe: v })} size="sm" />
    </Section>
  ) : null;

  return (
    <View>
      {!compact ? (
        <View style={styles.descRow}>
          <Text style={styles.descIcon}>{meta.icon}</Text>
          <View style={{ flex: 1 }}>
            <Text style={styles.descLabel}>{meta.label}</Text>
            <Text style={styles.descText}>{meta.desc}</Text>
          </View>
        </View>
      ) : null}
      {TF}
      {renderKnobs(type, cfg, set, compact)}
    </View>
  );
}

function renderKnobs(
  type: SignalType,
  cfg: Record<string, any>,
  set: (p: Record<string, any>) => void,
  compact: boolean,
) {
  switch (type) {
    case 'rsi': {
      const condOpts: ChipOption<string>[] = [
        { value: 'lt', label: 'Oversold (<30)' },
        { value: 'gt', label: 'Overbought (>70)' },
        { value: 'rising', label: 'Rising' },
        { value: 'falling', label: 'Falling' },
        { value: 'midline_cross_up', label: 'Cross above 50' },
        { value: 'midline_cross_down', label: 'Cross below 50' },
        { value: 'custom', label: 'Custom level' },
      ];
      const cond = cfg.condition || 'lt';
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={cond} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          {cond === 'custom' ? (
            <Stepper
              label="RSI level"
              value={cfg.value ?? 50}
              onChange={(v) => set({ value: v })}
              min={5} max={95} step={1}
              presets={[20, 30, 40, 50, 60, 70, 80]}
            />
          ) : null}
        </View>
      );
    }

    case 'macd': {
      const opts: ChipOption<string>[] = [
        { value: 'bullish_cross', label: 'Bullish cross' },
        { value: 'bearish_cross', label: 'Bearish cross' },
        { value: 'bullish',       label: 'Above signal' },
        { value: 'bearish',       label: 'Below signal' },
      ];
      return (
        <Section label="Condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'bullish_cross'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'ema': {
      const condOpts: ChipOption<string>[] = [
        { value: 'golden_cross', label: 'Golden cross' },
        { value: 'death_cross',  label: 'Death cross' },
        { value: 'bullish',      label: 'Fast > slow' },
        { value: 'bearish',      label: 'Fast < slow' },
      ];
      const periodOpts: ChipOption<string>[] = [
        { value: '9/21',   label: '9/21' },
        { value: '20/50',  label: '20/50' },
        { value: '50/200', label: '50/200' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={cfg.condition || 'golden_cross'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Section label="EMA periods" compact={compact}>
            <ChipRow options={periodOpts} value={cfg.periods || '9/21'} onChange={(v) => set({ periods: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'bb': {
      const opts: ChipOption<string>[] = [
        { value: 'squeeze',          label: 'Squeeze (low vol)' },
        { value: 'above_upper',      label: 'Above upper band' },
        { value: 'below_lower',      label: 'Below lower band' },
        { value: 'upper_touch',      label: 'Touched upper' },
        { value: 'lower_touch',      label: 'Touched lower' },
        { value: 'mean_reversion',   label: 'Returning to mean' },
      ];
      return (
        <Section label="Condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'squeeze'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'stoch_rsi': {
      const opts: ChipOption<string>[] = [
        { value: 'oversold',       label: 'Oversold' },
        { value: 'overbought',     label: 'Overbought' },
        { value: 'bullish_cross',  label: 'Bullish cross' },
        { value: 'bearish_cross',  label: 'Bearish cross' },
      ];
      return (
        <Section label="Condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'oversold'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'volume_spike':
      return (
        <Stepper
          label="Volume × 20-period avg"
          value={cfg.multiplier ?? 2}
          onChange={(v) => set({ multiplier: v })}
          min={1.2} max={20} step={0.5} decimals={1}
          unit="×"
          presets={[1.5, 2, 3, 5, 10]}
          hint="Higher = rarer, more meaningful spike"
        />
      );

    case 'price_momentum': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'up',     label: '🚀 Pumped' },
        { value: 'down',   label: '📉 Dropped' },
        { value: 'either', label: '↕ Either' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts} value={cfg.pm_dir || 'up'} onChange={(v) => set({ pm_dir: v })} size="sm" />
          </Section>
          <Stepper
            label="Min % move"
            value={cfg.pm_pct ?? 10}
            onChange={(v) => set({ pm_pct: v })}
            min={1} max={50} step={1} unit="%"
            presets={[3, 5, 10, 15, 20]}
          />
          <Stepper
            label="Within (minutes)"
            value={cfg.pm_window ?? 15}
            onChange={(v) => set({ pm_window: v })}
            min={1} max={240} step={5}
            presets={[5, 15, 30, 60, 120]}
          />
        </View>
      );
    }

    case 'breakout': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'up',     label: '⬆ Up' },
        { value: 'down',   label: '⬇ Down' },
        { value: 'either', label: '↕ Either' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts} value={cfg.bo_dir || 'up'} onChange={(v) => set({ bo_dir: v })} size="sm" />
          </Section>
          <Stepper
            label="Range lookback (candles)"
            value={cfg.bo_lookback ?? 20}
            onChange={(v) => set({ bo_lookback: v })}
            min={5} max={200} step={5}
            presets={[10, 20, 50, 100]}
          />
          <Stepper
            label="Min breakout %"
            value={cfg.bo_pct ?? 1}
            onChange={(v) => set({ bo_pct: v })}
            min={0.1} max={10} step={0.1} unit="%" decimals={1}
            presets={[0.5, 1, 2, 3]}
          />
        </View>
      );
    }

    case 'support_resistance': {
      const opts: ChipOption<string>[] = [
        { value: 'at_support',       label: 'At support' },
        { value: 'at_resistance',    label: 'At resistance' },
        { value: 'breakout_above',   label: 'Breaks resistance' },
        { value: 'breakout_below',   label: 'Breaks support' },
      ];
      return (
        <Section label="Condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'at_support'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'vwap_deviation': {
      const sideOpts: ChipOption<string>[] = [
        { value: 'below', label: 'Below VWAP' },
        { value: 'above', label: 'Above VWAP' },
      ];
      return (
        <View>
          <Section label="Side" compact={compact}>
            <ChipRow options={sideOpts} value={cfg.vwap_side || 'below'} onChange={(v) => set({ vwap_side: v })} size="sm" />
          </Section>
          <Stepper
            label="Distance from VWAP"
            value={cfg.vwap_pct ?? 3}
            onChange={(v) => set({ vwap_pct: v })}
            min={0.5} max={20} step={0.5} unit="%" decimals={1}
            presets={[1, 2, 3, 5, 8]}
          />
        </View>
      );
    }

    case 'candlestick': {
      const PATTERNS: ChipOption<string>[] = [
        { value: 'bullish_engulfing', label: '🟢 Bullish engulfing' },
        { value: 'bearish_engulfing', label: '🔴 Bearish engulfing' },
        { value: 'hammer',            label: '🔨 Hammer' },
        { value: 'shooting_star',     label: '⭐ Shooting star' },
        { value: 'doji',              label: '✚ Doji' },
        { value: 'morning_star',      label: '🌅 Morning star' },
        { value: 'evening_star',      label: '🌆 Evening star' },
        { value: 'three_white_soldiers', label: '👍 3 white soldiers' },
        { value: 'three_black_crows',    label: '👎 3 black crows' },
        { value: 'pin_bar',           label: '📍 Pin bar' },
      ];
      return (
        <Section label="Pattern" compact={compact}>
          <ChipRow options={PATTERNS} value={cfg.pattern || 'bullish_engulfing'} onChange={(v) => set({ pattern: v })} size="sm" />
        </Section>
      );
    }

    case 'order_block': {
      const opts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish OB (demand)' },
        { value: 'bearish', label: '🔴 Bearish OB (supply)' },
      ];
      return (
        <Section label="Block type" compact={compact}>
          <ChipRow options={opts} value={cfg.ob_type || 'bullish'} onChange={(v) => set({ ob_type: v })} size="sm" />
        </Section>
      );
    }

    case 'fvg': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish FVG' },
        { value: 'bearish', label: '🔴 Bearish FVG' },
      ];
      const condOpts: ChipOption<string>[] = [
        { value: 'gap_exists',     label: 'Gap exists' },
        { value: 'just_formed',    label: 'Just formed' },
        { value: 'price_in_gap',   label: 'Price inside' },
        { value: 'tap_and_reject', label: 'Tap & reject' },
        { value: 'approaching',    label: 'Approaching' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts} value={cfg.fvg_dir || 'bullish'} onChange={(v) => set({ fvg_dir: v })} size="sm" />
          </Section>
          <Section label="Trigger" compact={compact}>
            <ChipRow options={condOpts} value={cfg.condition || 'gap_exists'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Min gap %"
            value={cfg.min_gap_pct ?? 0.3}
            onChange={(v) => set({ min_gap_pct: v })}
            min={0} max={5} step={0.1} unit="%" decimals={1}
            presets={[0, 0.2, 0.5, 1]}
          />
        </View>
      );
    }

    case 'market_structure': {
      const opts: ChipOption<string>[] = [
        { value: 'bos_bullish',   label: '⬆ Bullish BOS' },
        { value: 'bos_bearish',   label: '⬇ Bearish BOS' },
        { value: 'choch_bullish', label: '🔄 Bullish CHoCH' },
        { value: 'choch_bearish', label: '🔃 Bearish CHoCH' },
      ];
      return (
        <Section label="Condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'bos_bullish'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'adx_filter': {
      const opts: ChipOption<string>[] = [
        { value: 'ranging',      label: 'Ranging (ADX<20)' },
        { value: 'trending',     label: 'Trending (ADX>25)' },
        { value: 'strong_trend', label: 'Strong trend (>40)' },
        { value: 'weak',         label: 'Weak / no trend' },
      ];
      return (
        <Section label="Market regime" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'ranging'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'sma': {
      const condOpts: ChipOption<string>[] = [
        { value: 'price_above',    label: 'Price above SMA ↑' },
        { value: 'price_below',    label: 'Price below SMA ↓' },
        { value: 'above_ribbon',   label: 'Above full ribbon 🟢' },
        { value: 'below_ribbon',   label: 'Below full ribbon 🔴' },
        { value: 'inside_ribbon',  label: 'Inside ribbon (neutral)' },
      ];
      const srcOpts: ChipOption<string>[] = [
        { value: 'close', label: 'Close' },
        { value: 'high',  label: 'High' },
        { value: 'low',   label: 'Low' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={cfg.condition || 'price_above'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Section label="Source" compact={compact}>
            <ChipRow options={srcOpts} value={cfg.source || 'close'} onChange={(v) => set({ source: v })} size="sm" />
          </Section>
          <Stepper
            label="SMA period"
            value={cfg.period ?? 200}
            onChange={(v) => set({ period: v })}
            min={5} max={500} step={5}
            presets={[20, 50, 100, 200]}
            hint="200 SMA on close = classic long-term trend filter"
          />
        </View>
      );
    }

    case 'supertrend': {
      const opts: ChipOption<string>[] = [
        { value: 'bullish_flip', label: 'Flipped bullish ↑' },
        { value: 'bearish_flip', label: 'Flipped bearish ↓' },
        { value: 'bullish',      label: 'Is bullish' },
        { value: 'bearish',      label: 'Is bearish' },
      ];
      return (
        <Section label="Signal" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'bullish_flip'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'ichimoku': {
      const opts: ChipOption<string>[] = [
        { value: 'above_cloud',        label: 'Price above Cloud' },
        { value: 'below_cloud',        label: 'Price below Cloud' },
        { value: 'tk_cross_up',        label: 'TK cross up' },
        { value: 'tk_cross_down',      label: 'TK cross down' },
        { value: 'kumo_breakout_up',   label: 'Kumo breakout up' },
        { value: 'kumo_breakout_down', label: 'Kumo breakout down' },
      ];
      return (
        <Section label="Ichimoku signal" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'above_cloud'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'donchian': {
      const opts: ChipOption<string>[] = [
        { value: 'upper_break', label: 'Breaks upper band ↑' },
        { value: 'lower_break', label: 'Breaks lower band ↓' },
        { value: 'near_upper',  label: 'Near upper (overbought)' },
        { value: 'near_lower',  label: 'Near lower (oversold)' },
      ];
      return (
        <View>
          <Section label="Donchian signal" compact={compact}>
            <ChipRow options={opts} value={cfg.condition || 'upper_break'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Channel period"
            value={cfg.period ?? 20}
            onChange={(v) => set({ period: v })}
            min={5} max={100} step={1}
            presets={[10, 20, 50, 100]}
          />
        </View>
      );
    }

    case 'cci': {
      const condOpts: ChipOption<string>[] = [
        { value: 'bullish',    label: 'Trend up (CCI > 0)' },
        { value: 'bearish',    label: 'Trend down (CCI < 0)' },
        { value: 'oversold',   label: 'Oversold (< -100)' },
        { value: 'overbought', label: 'Overbought (> +100)' },
      ];
      const maOpts: ChipOption<string>[] = [
        { value: 'none', label: 'None (raw)' },
        { value: 'SMA',  label: 'SMA' },
        { value: 'EMA',  label: 'EMA' },
        { value: 'SMMA', label: 'SMMA' },
        { value: 'WMA',  label: 'WMA' },
        { value: 'VWMA', label: 'VWMA' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={cfg.condition || 'bullish'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="CCI period"
            value={cfg.period ?? 20}
            onChange={(v) => set({ period: v })}
            min={5} max={100} step={1}
            presets={[10, 14, 20, 50]}
          />
          <Section label="MA smoothing (optional)" compact={compact}>
            <ChipRow options={maOpts} value={cfg.ma_type || 'none'} onChange={(v) => set({ ma_type: v })} size="sm" />
          </Section>
          {cfg.ma_type && cfg.ma_type !== 'none' ? (
            <Stepper
              label="Smoothing period"
              value={cfg.ma_period ?? 3}
              onChange={(v) => set({ ma_period: v })}
              min={2} max={20} step={1}
              presets={[3, 5, 9, 14]}
            />
          ) : null}
        </View>
      );
    }

    case 'mfi': {
      const opts: ChipOption<string>[] = [
        { value: 'oversold',   label: 'Oversold (< 20)' },
        { value: 'overbought', label: 'Overbought (> 80)' },
        { value: 'rising',     label: 'Rising (inflow)' },
        { value: 'falling',    label: 'Falling (outflow)' },
      ];
      return (
        <Section label="MFI condition" compact={compact}>
          <ChipRow options={opts} value={cfg.condition || 'oversold'} onChange={(v) => set({ condition: v })} size="sm" />
        </Section>
      );
    }

    case 'roc': {
      const opts: ChipOption<string>[] = [
        { value: 'positive',     label: 'ROC positive' },
        { value: 'negative',     label: 'ROC negative' },
        { value: 'cross_up',     label: 'Crossed above 0' },
        { value: 'cross_down',   label: 'Crossed below 0' },
        { value: 'accelerating', label: 'Accelerating' },
      ];
      return (
        <View>
          <Section label="ROC condition" compact={compact}>
            <ChipRow options={opts} value={cfg.condition || 'cross_up'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Lookback period"
            value={cfg.period ?? 10}
            onChange={(v) => set({ period: v })}
            min={2} max={100} step={1}
            presets={[5, 10, 20, 50]}
          />
        </View>
      );
    }

    case 'divergence': {
      // Backend `eval_divergence` only supports RSI/MACD + bullish/bearish.
      // Hidden divergences and stoch_rsi/cci/obv are evaluator-unimplemented,
      // so we trim the UI to evaluator-safe options to keep saved strategies firing.
      const indOpts: ChipOption<string>[] = [
        { value: 'rsi',  label: 'RSI' },
        { value: 'macd', label: 'MACD' },
      ];
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish' },
        { value: 'bearish', label: '🔴 Bearish' },
      ];
      return (
        <View>
          <Section label="Indicator" compact={compact}>
            <ChipRow options={indOpts} value={cfg.indicator || 'rsi'} onChange={(v) => set({ indicator: v })} size="sm" />
          </Section>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts} value={cfg.direction || 'bullish'} onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'tradingview_webhook':
      return null;

    case 'forex_liquidity_pa': {
      const patternOpts: ChipOption<string>[] = [
        { value: 'sweep_eqh',    label: '⬆️ Sweep eq-highs' },
        { value: 'sweep_eql',    label: '⬇️ Sweep eq-lows' },
        { value: 'pin_bar_bull', label: '🟢 Bullish pin' },
        { value: 'pin_bar_bear', label: '🔴 Bearish pin' },
        { value: 'engulf_bull',  label: '🟢 Bull engulf' },
        { value: 'engulf_bear',  label: '🔴 Bear engulf' },
        { value: 'inside_bar',   label: '📦 Inside bar' },
      ];
      const pattern = (cfg.pattern as string) || 'sweep_eqh';
      const isSweep = pattern === 'sweep_eqh' || pattern === 'sweep_eql';
      return (
        <View>
          <Section label="Pattern" compact={compact}>
            <ChipRow options={patternOpts} value={pattern} onChange={(v) => set({ pattern: v })} size="sm" />
          </Section>
          {isSweep ? (
            <>
              <Stepper label="Lookback bars" value={cfg.lookback ?? 20}
                min={5} max={100} step={1}
                onChange={(v) => set({ lookback: v })} />
              <Stepper label="Tolerance (pips)" value={cfg.tolerance_pips ?? 3}
                min={0.5} max={20} step={0.5} decimals={1} unit="pips"
                onChange={(v) => set({ tolerance_pips: v })} />
            </>
          ) : null}
        </View>
      );
    }

    default:
      return null;
  }
}

function Section({ label, children, compact }: { label: string; children: React.ReactNode; compact: boolean }) {
  return (
    <View style={[styles.section, compact && { marginTop: 6 }]}>
      <Text style={styles.sectionLabel}>{label}</Text>
      {children}
    </View>
  );
}

const styles = StyleSheet.create({
  descRow: {
    flexDirection: 'row', alignItems: 'flex-start', gap: 10,
    backgroundColor: colors.bgElev,
    borderRadius: radius.md,
    padding: 10,
    marginBottom: spacing.sm,
    borderWidth: 1, borderColor: colors.border,
  },
  descIcon:  { fontSize: 20, marginTop: 1 },
  descLabel: { fontFamily: font.semibold, fontSize: 13.5, color: colors.text },
  descText:  { fontFamily: font.regular, fontSize: 12, color: colors.textMute, marginTop: 2, lineHeight: 16 },
  section:       { marginTop: spacing.sm },
  sectionLabel:  {
    fontFamily: font.medium, fontSize: 11, color: colors.textDim,
    textTransform: 'uppercase', letterSpacing: 0.8, marginBottom: 6,
  },
});
