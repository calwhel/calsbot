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

    case 'fvg':
    case 'ifvg': {
      const isIfvg = type === 'ifvg';
      const dirOpts: ChipOption<string>[] = isIfvg ? [
        { value: 'bullish', label: '🟢 Bullish IFVG (price re-enters bearish gap)' },
        { value: 'bearish', label: '🔴 Bearish IFVG (price re-enters bullish gap)' },
      ] : [
        { value: 'bullish', label: '🟢 Bullish FVG (buy-side imbalance)' },
        { value: 'bearish', label: '🔴 Bearish FVG (sell-side imbalance)' },
      ];
      const condOpts: ChipOption<string>[] = isIfvg ? [
        { value: 'price_in_gap',   label: 'Price re-enters gap' },
        { value: 'tap_and_reject', label: 'Tap & reject (reversal)' },
        { value: 'approaching',    label: 'Approaching gap' },
        { value: 'gap_exists',     label: 'Gap still open' },
      ] : [
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
            <ChipRow options={condOpts} value={cfg.condition || (isIfvg ? 'price_in_gap' : 'gap_exists')} onChange={(v) => set({ condition: v })} size="sm" />
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

    case 'forex_news_avoidance': {
      const impactOpts: ChipOption<string>[] = [
        { value: 'high',   label: '🔴 High only' },
        { value: 'medium', label: '🟡 Medium & above' },
        { value: 'low',    label: '⚪ All events' },
      ];
      return (
        <View>
          <Section label="Minimum impact" compact={compact}>
            <ChipRow options={impactOpts}
              value={(cfg.min_impact as string) || 'high'}
              onChange={(v) => set({ min_impact: v })} size="sm" />
          </Section>
          <Stepper label="Block before event" value={cfg.minutes_before ?? 30}
            min={5} max={120} step={5} unit="min"
            onChange={(v) => set({ minutes_before: v })} />
          <Stepper label="Block after event" value={cfg.minutes_after ?? 30}
            min={5} max={120} step={5} unit="min"
            onChange={(v) => set({ minutes_after: v })} />
        </View>
      );
    }

    case 'forex_currency_strength': {
      const winOpts: ChipOption<string>[] = [
        { value: '1h', label: '1h' },
        { value: '4h', label: '4h' },
        { value: '1d', label: '1d' },
      ];
      const dirOpts: ChipOption<string>[] = [
        { value: 'either',       label: '↕ Either side' },
        { value: 'base_strong',  label: '⬆ Base stronger (LONG)' },
        { value: 'quote_strong', label: '⬇ Quote stronger (SHORT)' },
      ];
      return (
        <View>
          <Section label="Strength window" compact={compact}>
            <ChipRow options={winOpts}
              value={(cfg.window as string) || '4h'}
              onChange={(v) => set({ window: v })} size="sm" />
          </Section>
          <Section label="Direction bias" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'either'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper label="Min differential score" value={cfg.min_diff ?? 0.6}
            min={0.1} max={3.0} step={0.1} decimals={1}
            onChange={(v) => set({ min_diff: v })} />
        </View>
      );
    }

    case 'stock_earnings_avoidance': {
      const modeOpts: ChipOption<string>[] = [
        { value: 'both',   label: '🛑 Block before & after' },
        { value: 'before', label: '⏪ Block only before (catch post-momentum)' },
        { value: 'after',  label: '⏩ Block only after (catch run-up)' },
      ];
      return (
        <View>
          <Stepper label="Days before earnings" value={cfg.days_before ?? 2}
            min={0} max={14} step={1} unit="d"
            onChange={(v) => set({ days_before: v })} />
          <Stepper label="Days after earnings" value={cfg.days_after ?? 1}
            min={0} max={14} step={1} unit="d"
            onChange={(v) => set({ days_after: v })} />
          <Section label="Blackout sides" compact={compact}>
            <ChipRow options={modeOpts}
              value={(cfg.mode as string) || 'both'}
              onChange={(v) => set({ mode: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'forex_cot': {
      const condOpts: ChipOption<string>[] = [
        { value: 'specs_extreme_long',  label: '🔴 Specs net-long extreme' },
        { value: 'specs_extreme_short', label: '🟢 Specs net-short extreme' },
        { value: 'specs_flipped_long',  label: '🟢 Specs flipped long' },
        { value: 'specs_flipped_short', label: '🔴 Specs flipped short' },
        { value: 'comm_extreme_long',   label: '🟢 Commercials net-long extreme' },
        { value: 'comm_extreme_short',  label: '🔴 Commercials net-short extreme' },
      ];
      const invOpts: ChipOption<string>[] = [
        { value: 'auto', label: 'Auto-flip for USD-base pairs' },
        { value: 'raw',  label: 'Raw (read non-USD leg directly)' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={(cfg.condition as string) || 'specs_extreme_long'}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper label="Percentile threshold" value={cfg.extreme_pct ?? 75}
            min={50} max={95} step={5} unit="th"
            onChange={(v) => set({ extreme_pct: v })} />
          <Stepper label="History window (weeks)" value={cfg.lookback_weeks ?? 52}
            min={8} max={156} step={4} unit="wk"
            onChange={(v) => set({ lookback_weeks: v })} />
          <Section label="Pair-direction handling" compact={compact}>
            <ChipRow options={invOpts}
              value={cfg.respect_pair_inversion === false ? 'raw' : 'auto'}
              onChange={(v) => set({ respect_pair_inversion: v === 'auto' })} size="sm" />
          </Section>
        </View>
      );
    }

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

    case 'session_level': {
      const refOpts: ChipOption<string>[] = [
        { value: 'session_low',  label: '⬇️ Session low' },
        { value: 'session_high', label: '⬆️ Session high' },
        { value: 'session_open', label: '🟡 Session open' },
        { value: 'daily_open',   label: '🔵 Daily open' },
      ];
      const condOpts: ChipOption<string>[] = [
        { value: 'near',  label: '🎯 Near level' },
        { value: 'above', label: '⬆️ Price above' },
        { value: 'below', label: '⬇️ Price below' },
      ];
      return (
        <View>
          <Section label="Reference level" compact={compact}>
            <ChipRow options={refOpts} value={(cfg.reference as string) || 'session_low'} onChange={(v) => set({ reference: v })} size="sm" />
          </Section>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={(cfg.condition as string) || 'near'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Nearness threshold"
            value={cfg.threshold_pct ?? 2}
            onChange={(v) => set({ threshold_pct: v })}
            min={0.5} max={10} step={0.5} unit="%" decimals={1}
            presets={[0.5, 1, 2, 3, 5]}
          />
        </View>
      );
    }

    case 'pivot_points': {
      const levelOpts: ChipOption<string>[] = [
        { value: 'pp', label: '◉ PP' },
        { value: 'r1', label: '🟢 R1' },
        { value: 'r2', label: '🟢 R2' },
        { value: 'r3', label: '🟢 R3' },
        { value: 's1', label: '🔴 S1' },
        { value: 's2', label: '🔴 S2' },
        { value: 's3', label: '🔴 S3' },
      ];
      const condOpts: ChipOption<string>[] = [
        { value: 'near',  label: '🎯 Near' },
        { value: 'above', label: '⬆️ Above' },
        { value: 'below', label: '⬇️ Below' },
      ];
      return (
        <View>
          <Section label="Pivot level" compact={compact}>
            <ChipRow options={levelOpts} value={(cfg.level as string) || 'r1'} onChange={(v) => set({ level: v })} size="sm" />
          </Section>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts} value={(cfg.condition as string) || 'near'} onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Tolerance (near only)"
            value={cfg.tolerance_pct ?? 0.3}
            onChange={(v) => set({ tolerance_pct: v })}
            min={0.1} max={2} step={0.1} unit="%" decimals={1}
            presets={[0.1, 0.2, 0.3, 0.5, 1]}
          />
        </View>
      );
    }

    case 'fx_killzone': {
      const kzOpts: ChipOption<string>[] = [
        { value: 'london_kz', label: '🇬🇧 London (07–09 UTC)' },
        { value: 'ny_kz',     label: '🇺🇸 NY (12–14 UTC)' },
        { value: 'asian_kz',  label: '🌏 Asian (20–23 UTC)' },
        { value: 'any_kz',    label: '🎯 Any killzone' },
      ];
      return (
        <Section label="Killzone window" compact={compact}>
          <ChipRow options={kzOpts}
            value={(cfg.killzone as string) || 'london_kz'}
            onChange={(v) => set({ killzone: v })} size="sm" />
        </Section>
      );
    }

    case 'fx_ote': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish OTE (demand zone)' },
        { value: 'bearish', label: '🔴 Bearish OTE (supply zone)' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'bullish'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper label="Swing lookback (bars)" value={cfg.swing_lookback ?? 20}
            min={5} max={100} step={5}
            onChange={(v) => set({ swing_lookback: v })}
            hint="Look back N bars to find the swing high/low that defines the retracement" />
          <Stepper label="Fib low (%)" value={cfg.fib_low ?? 61.8}
            min={50} max={75} step={0.1} decimals={1}
            onChange={(v) => set({ fib_low: v })} />
          <Stepper label="Fib high (%)" value={cfg.fib_high ?? 78.6}
            min={60} max={90} step={0.1} decimals={1}
            onChange={(v) => set({ fib_high: v })} />
        </View>
      );
    }

    case 'fx_displacement': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish (big green candle)' },
        { value: 'bearish', label: '🔴 Bearish (big red candle)' },
        { value: 'any',     label: '⚡ Either direction' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'any'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper label="Min body size (× avg range)" value={cfg.min_body_ratio ?? 3}
            min={1.5} max={8} step={0.5} decimals={1}
            onChange={(v) => set({ min_body_ratio: v })}
            hint="Candle body must be this many times larger than the 14-period average body" />
        </View>
      );
    }

    case 'fx_equal_hl': {
      const typeOpts: ChipOption<string>[] = [
        { value: 'eqh', label: '⬆️ Equal Highs (EQH)' },
        { value: 'eql', label: '⬇️ Equal Lows (EQL)' },
      ];
      return (
        <View>
          <Section label="Pattern" compact={compact}>
            <ChipRow options={typeOpts}
              value={(cfg.type as string) || 'eqh'}
              onChange={(v) => set({ type: v })} size="sm" />
          </Section>
          <Stepper label="Lookback (bars)" value={cfg.lookback ?? 30}
            min={10} max={100} step={5}
            onChange={(v) => set({ lookback: v })} />
          <Stepper label="Tolerance (pips)" value={cfg.tolerance_pips ?? 3}
            min={0.5} max={20} step={0.5} decimals={1} unit=" pips"
            onChange={(v) => set({ tolerance_pips: v })}
            hint="Two highs/lows within this pip distance count as 'equal' — a liquidity pool" />
        </View>
      );
    }

    case 'fx_breaker': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish breaker (old supply → support)' },
        { value: 'bearish', label: '🔴 Bearish breaker (old demand → resistance)' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'bullish'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper label="Lookback (bars)" value={cfg.lookback ?? 50}
            min={20} max={200} step={10}
            onChange={(v) => set({ lookback: v })} />
          <Stepper label="Zone tolerance (%)" value={cfg.tolerance_pct ?? 0.5}
            min={0.1} max={3} step={0.1} decimals={1} unit="%"
            onChange={(v) => set({ tolerance_pct: v })}
            hint="Price must return within this % of the breaker zone to trigger" />
        </View>
      );
    }

    case 'fx_cisd': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish (sellers done → buyers in)' },
        { value: 'bearish', label: '🔴 Bearish (buyers done → sellers in)' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'bullish'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper label="Max delivery run (candles)" value={cfg.max_run ?? 10}
            min={1} max={50} step={1}
            onChange={(v) => set({ max_run: v })}
            hint="How many opposing candles back to look for the delivery origin to close through" />
        </View>
      );
    }

    case 'fx_pd_array': {
      const biasOpts: ChipOption<string>[] = [
        { value: 'discount', label: '🟢 Discount (below 50% — buy zone)' },
        { value: 'premium',  label: '🔴 Premium (above 50% — sell zone)' },
      ];
      return (
        <View>
          <Section label="Bias" compact={compact}>
            <ChipRow options={biasOpts}
              value={(cfg.bias as string) || 'discount'}
              onChange={(v) => set({ bias: v })} size="sm" />
          </Section>
          <Stepper label="Swing lookback (bars)" value={cfg.lookback ?? 50}
            min={20} max={200} step={10}
            onChange={(v) => set({ lookback: v })}
            hint="Bars to scan for the recent swing H/L" />
        </View>
      );
    }

    case 'fx_judas_swing': {
      const sessOpts: ChipOption<string>[] = [
        { value: 'london', label: '🇬🇧 London (08:00 UTC)' },
        { value: 'ny',     label: '🗽 NY (13:30 UTC)' },
      ];
      return (
        <View>
          <Section label="Session" compact={compact}>
            <ChipRow options={sessOpts}
              value={(cfg.session as string) || 'london'}
              onChange={(v) => set({ session: v })} size="sm" />
          </Section>
          <Stepper label="Min fake sweep (pips)" value={cfg.swing_pips ?? 10}
            min={3} max={50} step={1}
            onChange={(v) => set({ swing_pips: v })}
            hint="Minimum pip distance the fake move must extend beyond prior range" />
          <Stepper label="Min reversal (pips)" value={cfg.reversal_pips ?? 5}
            min={2} max={30} step={1}
            onChange={(v) => set({ reversal_pips: v })}
            hint="Price must then reverse this many pips for confirmation" />
        </View>
      );
    }

    case 'fx_silver_bullet': {
      const winOpts: ChipOption<string>[] = [
        { value: 'any',      label: '🔁 Any window' },
        { value: 'early_am', label: '🌙 03:00–04:00 NY' },
        { value: 'am',       label: '🌅 10:00–11:00 NY' },
        { value: 'pm',       label: '🌇 15:00–16:00 NY' },
      ];
      return (
        <View>
          <Section label="NY time window" compact={compact}>
            <ChipRow options={winOpts}
              value={(cfg.window as string) || 'any'}
              onChange={(v) => set({ window: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'opening_range_break': {
      const sessOpts: ChipOption<string>[] = [
        { value: 'london',   label: '🇬🇧 London 08:00' },
        { value: 'ny',       label: '🗽 NY 13:30' },
        { value: 'asia',     label: '🌏 Asia 00:00' },
        { value: 'midnight', label: '🕛 Midnight UTC' },
      ];
      const minsOpts: ChipOption<string>[] = [
        { value: '5',  label: '5m' },
        { value: '15', label: '15m' },
        { value: '30', label: '30m' },
        { value: '60', label: '1h' },
      ];
      const dirOpts: ChipOption<string>[] = [
        { value: 'both', label: '↕ Either' },
        { value: 'up',   label: '↑ Up only' },
        { value: 'down', label: '↓ Down only' },
      ];
      return (
        <View>
          <Section label="Session start" compact={compact}>
            <ChipRow options={sessOpts}
              value={(cfg.session_start as string) || 'london'}
              onChange={(v) => set({ session_start: v })} size="sm" />
          </Section>
          <Section label="ORB window" compact={compact}>
            <ChipRow options={minsOpts}
              value={String(cfg.orb_minutes ?? 30)}
              onChange={(v) => set({ orb_minutes: Number(v) })} size="sm" />
          </Section>
          <Section label="Break direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'both'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'vwap_cross': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'cross_above', label: '↑ Cross above VWAP (bullish)' },
        { value: 'cross_below', label: '↓ Cross below VWAP (bearish)' },
      ];
      return (
        <View>
          <Section label="Direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'cross_above'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
        </View>
      );
    }

    case 'stochastic': {
      const condOpts: ChipOption<string>[] = [
        { value: 'oversold',      label: '📉 Oversold <20' },
        { value: 'overbought',    label: '📈 Overbought >80' },
        { value: 'bullish_cross', label: '↑ %K crosses %D (bullish)' },
        { value: 'bearish_cross', label: '↓ %K crosses %D (bearish)' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={(cfg.condition as string) || 'bullish_cross'}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="%K period"
            value={cfg.k_period ?? 14}
            onChange={(v) => set({ k_period: v })}
            min={5} max={30} step={1}
            presets={[5, 9, 14, 21]}
            hint="Lookback for highest high / lowest low"
          />
          <Stepper
            label="%D smoothing"
            value={cfg.d_period ?? 3}
            onChange={(v) => set({ d_period: v })}
            min={1} max={9} step={1}
            presets={[1, 3, 5]}
            hint="SMA of %K — signal line"
          />
        </View>
      );
    }

    case 'fx_po3': {
      const dirOpts: ChipOption<string>[] = [
        { value: 'bullish', label: '🟢 Bullish (swept lows → up)' },
        { value: 'bearish', label: '🔴 Bearish (swept highs → down)' },
      ];
      return (
        <View>
          <Section label="Distribution direction" compact={compact}>
            <ChipRow options={dirOpts}
              value={(cfg.direction as string) || 'bullish'}
              onChange={(v) => set({ direction: v })} size="sm" />
          </Section>
          <Stepper
            label="Min manipulation sweep (pips)"
            value={cfg.sweep_pips ?? 5}
            onChange={(v) => set({ sweep_pips: v })}
            min={2} max={30} step={1} unit=" pips" decimals={0}
            presets={[3, 5, 8, 12]}
            hint="How far price must sweep beyond the Asian range to count as manipulation"
          />
        </View>
      );
    }

    case 'wyckoff': {
      const phaseOpts: ChipOption<string>[] = [
        { value: 'spring',    label: '🌱 Spring (bullish)' },
        { value: 'upthrust',  label: '🏹 Upthrust (bearish)' },
        { value: 'test',      label: '🔍 Low-vol test' },
        { value: 'markup',    label: '📈 Markup start' },
        { value: 'markdown',  label: '📉 Markdown start' },
      ];
      return (
        <View>
          <Section label="Wyckoff phase" compact={compact}>
            <ChipRow options={phaseOpts}
              value={(cfg.phase as string) || 'spring'}
              onChange={(v) => set({ phase: v })} size="sm" />
          </Section>
          <Stepper
            label="Range lookback (bars)"
            value={cfg.lookback ?? 30}
            onChange={(v) => set({ lookback: v })}
            min={10} max={100} step={5}
            presets={[20, 30, 50, 80]}
            hint="Bars used to define the trading range for support/resistance"
          />
        </View>
      );
    }

    case 'atr_filter': {
      const condOpts: ChipOption<string>[] = [
        { value: 'volatile',  label: '⚡ Volatile (ATR% ≥ min)' },
        { value: 'expanding', label: '📈 Expanding vs N bars ago' },
      ];
      const cond = (cfg.condition as string) || 'volatile';
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={cond}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="ATR period"
            value={cfg.period ?? 14}
            onChange={(v) => set({ period: v })}
            min={5} max={50} step={1}
            presets={[7, 14, 21]}
            hint="Lookback for the Average True Range"
          />
          {cond === 'volatile' ? (
            <Stepper
              label="Min ATR % of price"
              value={cfg.min_atr_pct ?? 0.3}
              onChange={(v) => set({ min_atr_pct: v })}
              min={0.05} max={5} step={0.05} unit="%" decimals={2}
              presets={[0.2, 0.3, 0.5, 1]}
              hint="Only trade when ATR is at least this % of price"
            />
          ) : (
            <Stepper
              label="Compare to N bars ago"
              value={cfg.lookback ?? 5}
              onChange={(v) => set({ lookback: v })}
              min={1} max={30} step={1}
              presets={[3, 5, 10]}
              hint="ATR must be higher than it was this many bars ago"
            />
          )}
        </View>
      );
    }

    case 'rvol': {
      const condOpts: ChipOption<string>[] = [
        { value: 'high', label: '📢 High (≥ threshold)' },
        { value: 'low',  label: '🔇 Low (< threshold)' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={(cfg.condition as string) || 'high'}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="RVOL threshold"
            value={cfg.threshold ?? 1.5}
            onChange={(v) => set({ threshold: v })}
            min={0.5} max={10} step={0.1} unit="×" decimals={1}
            presets={[1.2, 1.5, 2, 3]}
            hint="Current volume vs the average (1.5× = 50% above normal)"
          />
          <Stepper
            label="Average period"
            value={cfg.period ?? 20}
            onChange={(v) => set({ period: v })}
            min={5} max={100} step={1}
            presets={[10, 20, 50]}
            hint="Bars used for the average-volume baseline"
          />
        </View>
      );
    }

    case 'vwap_bands': {
      const condOpts: ChipOption<string>[] = [
        { value: 'below_lower', label: '📉 At/below lower band (long)' },
        { value: 'above_upper', label: '📈 At/above upper band (short)' },
        { value: 'inside',      label: '↔ Inside the bands' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={(cfg.condition as string) || 'below_lower'}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
          <Stepper
            label="Band width (std dev)"
            value={cfg.num_std ?? 2.0}
            onChange={(v) => set({ num_std: v })}
            min={0.5} max={4} step={0.1} unit="σ" decimals={1}
            presets={[1, 1.5, 2, 2.5]}
            hint="VWAP ± this many standard deviations"
          />
        </View>
      );
    }

    case 'vwap_bias': {
      const condOpts: ChipOption<string>[] = [
        { value: 'above', label: '🟢 Above VWAP (long bias)' },
        { value: 'below', label: '🔴 Below VWAP (short bias)' },
      ];
      return (
        <View>
          <Section label="Condition" compact={compact}>
            <ChipRow options={condOpts}
              value={(cfg.condition as string) || 'above'}
              onChange={(v) => set({ condition: v })} size="sm" />
          </Section>
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
