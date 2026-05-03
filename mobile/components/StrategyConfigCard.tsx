import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { InfoCard, InfoBullet } from '@/components/InfoCard';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * Renders a strategy's machine-readable `config` blob as a human-readable
 * "how this strategy works" panel. Accepts the loose `Record<string, unknown>`
 * shape returned by /api/strategies and is defensive about missing pieces — a
 * legacy strategy without an `entry_conditions` block still renders cleanly.
 */
type Cfg = Record<string, unknown>;

function asObj(v: unknown): Cfg | null {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Cfg) : null;
}
function asStr(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null;
}
function asNum(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.length > 0 && Number.isFinite(Number(v))) return Number(v);
  return null;
}
function asArr(v: unknown): unknown[] | null {
  return Array.isArray(v) ? v : null;
}

/** Convert a comparison operator string into plain English. */
function opPhrase(op: string, fallback = 'reaches'): string {
  switch (op) {
    case 'lt': case '<': return 'drops below';
    case 'gt': case '>': return 'climbs above';
    case 'lte': case '<=': return 'is at or below';
    case 'gte': case '>=': return 'is at or above';
    case 'eq': case '==': return 'equals';
    case 'cross_above': case 'crosses_above': return 'crosses above';
    case 'cross_below': case 'crosses_below': return 'crosses below';
    default: return op || fallback;
  }
}

/** Pretty name for an indicator-condition `name` field. */
function indicatorLabel(name: string): string {
  const n = name.toLowerCase();
  if (n === 'rsi') return 'RSI';
  if (n === 'macd') return 'MACD';
  if (n === 'ema') return 'EMA';
  if (n === 'sma') return 'SMA';
  if (n === 'sma_cross') return 'SMA cross';
  if (n === 'sma_ribbon') return 'SMA ribbon';
  if (n === 'ema_cross') return 'EMA cross';
  if (n === 'ema_ribbon') return 'EMA ribbon';
  if (n === 'bollinger' || n === 'bbands' || n === 'bb') return 'Bollinger Bands';
  if (n === 'stoch_rsi' || n === 'stochrsi') return 'Stoch RSI';
  if (n === 'stoch') return 'Stochastic';
  if (n === 'atr') return 'ATR';
  if (n === 'vwap') return 'VWAP';
  if (n === 'ichimoku') return 'Ichimoku';
  if (n === 'supertrend') return 'SuperTrend';
  if (n === 'adx') return 'ADX';
  if (n === 'cci') return 'CCI';
  if (n === 'obv') return 'OBV';
  if (n === 'mfi') return 'MFI';
  if (n === 'keltner') return 'Keltner Channel';
  if (n === 'donchian') return 'Donchian Channel';
  return name.toUpperCase();
}

/** Make snake_case condition keywords readable ("near_support" → "near support"). */
function humanise(s: string): string {
  return s.replace(/_/g, ' ').toLowerCase();
}

function tfSuffix(tf: string): string {
  return tf ? ` on the ${tf} timeframe` : '';
}

function withDir(dir: string): string {
  const d = dir.toUpperCase();
  return d === 'LONG' || d === 'SHORT' ? ` (${d})` : '';
}

/** Reference-frame label for the price_relative condition. */
function referenceLabel(ref: string): string {
  const r = ref.toLowerCase();
  if (r === 'daily_open') return "today's open";
  if (r === 'weekly_open') return "this week's open";
  if (r === 'session_high') return "today's session high";
  if (r === 'session_low') return "today's session low";
  return humanise(ref);
}

/**
 * Format an entry condition object into a plain-English sentence. Field names
 * and sub-condition values are aligned with the canonical backend schema in
 * `app/services/strategy_ta.py` (each `eval_*` function reveals the keys it
 * actually reads). Falls back to an honest "Custom <type> rule" line only when
 * the shape is too sparse to describe.
 */
function describeCondition(c: Cfg): string {
  const type = (asStr(c.type) || '').toLowerCase();
  const tf = asStr(c.timeframe) || '';
  const op = asStr(c.operator) || '';
  const val = asNum(c.value);
  const sub = asStr(c.condition) || '';
  const dir = asStr(c.direction) || '';

  // ── Indicator-typed conditions (RSI, MACD, EMA, Bollinger, …) ───────────
  if (type === 'indicator') {
    const name = asStr(c.name) || '';
    const label = indicatorLabel(name);
    if (sub) return `${label} signal: ${humanise(sub)}${tfSuffix(tf)}`;
    if (val != null) return `${label} ${opPhrase(op)} ${val}${tfSuffix(tf)}`;
    return `${label} signal triggers${tfSuffix(tf)}`;
  }

  // Bare-typed indicator aliases the evaluator wraps in `eval_indicator`.
  if (type === 'sma' || type === 'sma_cross' || type === 'sma_ribbon' || type === 'supertrend') {
    return `${indicatorLabel(type)} signal${sub ? `: ${humanise(sub)}` : ''}${tfSuffix(tf)}`;
  }

  // ── Price momentum (window_minutes + operator + value + direction) ──────
  if (type === 'price_momentum') {
    const windowMin = asNum(c.window_minutes);
    const reqDir = dir.toLowerCase(); // "up" | "down" | "any"
    const win = windowMin != null ? `${windowMin}-minute` : 'recent';
    const dirPart = reqDir === 'up' ? ' upward' : reqDir === 'down' ? ' downward' : '';
    if (val != null) return `Price moves${dirPart} ${opPhrase(op, 'by at least')} ${val}% over a ${win} window`;
    return `Strong${dirPart} price momentum over a ${win} window`;
  }

  // ── Volume spike (multiplier) ───────────────────────────────────────────
  if (type === 'volume_spike') {
    const mult = asNum(c.multiplier);
    const m = mult != null ? mult : 1.5;
    return `Volume spikes to ${m}× the recent average`;
  }

  // ── Support / resistance (condition: at_support|at_resistance|
  //    breakout_above|breakout_below|between) ─────────────────────────────
  if (type === 'support_resistance') {
    if (sub === 'at_support') return `Price tags a recent support level${tfSuffix(tf)}`;
    if (sub === 'at_resistance') return `Price tags a recent resistance level${tfSuffix(tf)}`;
    if (sub === 'breakout_above') return `Price breaks above resistance${tfSuffix(tf)}`;
    if (sub === 'breakout_below') return `Price breaks below support${tfSuffix(tf)}`;
    if (sub === 'between') return `Price trades inside a key support–resistance range${tfSuffix(tf)}`;
    return `Support/resistance level interaction${tfSuffix(tf)}`;
  }

  // ── Fair-value gap (canonical sub: gap_exists | just_formed |
  //    price_in_gap (default) | tap_and_reject | approaching | gap_filled) ──
  if (type === 'fvg') {
    const gapPctRaw = asNum(c.min_gap_pct);
    const atrMult = asNum(c.min_gap_atr_mult);
    const qualityBits: string[] = [];
    if (gapPctRaw != null && gapPctRaw > 0) qualityBits.push(`≥ ${gapPctRaw}% wide`);
    if (atrMult != null && atrMult > 0) qualityBits.push(`≥ ${atrMult}× ATR`);
    const qualityPart = qualityBits.length ? ` (${qualityBits.join(', ')})` : '';
    // FVG uses `direction` = "bullish" | "bearish" | "any" — surface the bias.
    const fvgDir = dir.toLowerCase();
    const biasWord = fvgDir === 'bullish' ? 'bullish ' : fvgDir === 'bearish' ? 'bearish ' : '';
    const tfPart = tfSuffix(tf);

    // Backend default when `condition` is omitted is `price_in_gap`, NOT
    // `gap_exists` — keep the empty/undefined branch aligned with that.
    const effectiveSub = sub || 'price_in_gap';

    if (effectiveSub === 'just_formed') return `A fresh ${biasWord}fair-value gap just formed${qualityPart}${tfPart}`;
    if (effectiveSub === 'price_in_gap' || effectiveSub === 'inside') return `Price is sitting inside a ${biasWord}fair-value gap${qualityPart}${tfPart}`;
    if (effectiveSub === 'tap_and_reject') return `Price wicks into a ${biasWord}fair-value gap and rejects out${qualityPart}${tfPart}`;
    if (effectiveSub === 'approaching') return `Price is approaching a ${biasWord}fair-value gap${qualityPart}${tfPart}`;
    if (effectiveSub === 'gap_filled') return `A ${biasWord}fair-value gap has been fully filled${qualityPart}${tfPart}`;
    if (effectiveSub === 'gap_exists') return `A qualifying ${biasWord}fair-value gap is present${qualityPart}${tfPart}`;
    return `${biasWord ? biasWord.charAt(0).toUpperCase() + biasWord.slice(1) : ''}Fair-value gap signal (${humanise(effectiveSub)})${qualityPart}${tfPart}`.trim();
  }

  // ── Candlestick pattern (uses `pattern`, NOT `condition`) ───────────────
  if (type === 'candlestick') {
    const pattern = asStr(c.pattern) || '';
    const label = pattern
      ? pattern.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase())
      : 'a key candlestick pattern';
    return `${pattern ? `Closes with a ${label}` : `Closes with ${label}`}${tfSuffix(tf)}`;
  }

  // ── Consecutive candles (uses `count` + `direction` red|green) ──────────
  if (type === 'consecutive_candles') {
    const count = asNum(c.count);
    const n = count != null ? count : 3;
    const colour = dir.toLowerCase() === 'green' ? 'green' : dir.toLowerCase() === 'red' ? 'red' : 'same-colour';
    return `${n} ${colour} candles close in a row${tfSuffix(tf)}`;
  }

  // ── Market structure (bos_bullish/bos_bearish/choch_bullish/choch_bearish) ─
  if (type === 'market_structure') {
    if (sub === 'bos_bullish') return `Bullish break of market structure (BOS)${tfSuffix(tf)}`;
    if (sub === 'bos_bearish') return `Bearish break of market structure (BOS)${tfSuffix(tf)}`;
    if (sub === 'choch_bullish') return `Bullish change-of-character (CHoCH) — uptrend resuming${tfSuffix(tf)}`;
    if (sub === 'choch_bearish') return `Bearish change-of-character (CHoCH) — downtrend resuming${tfSuffix(tf)}`;
    return `Market structure shift${withDir(dir)}${tfSuffix(tf)}`;
  }

  // ── Order block (uses `ob_type` or `direction`: bullish|bearish) ────────
  if (type === 'order_block') {
    const obType = (asStr(c.ob_type) || dir || 'bullish').toLowerCase();
    const side = obType === 'bearish' ? 'bearish' : 'bullish';
    return `Price reacts to a ${side} order block${tfSuffix(tf)}`;
  }

  // ── Fibonacci (level + condition: at_retracement | at_extension) ────────
  if (type === 'fibonacci') {
    const level = asNum(c.level);
    const lvl = level != null ? `${(level * 100).toFixed(1)}%` : 'a key';
    const kind = sub === 'at_extension' ? 'fib extension' : 'fib retracement';
    return `Price taps the ${lvl} ${kind}${tfSuffix(tf)}`;
  }

  // ── Divergence ──────────────────────────────────────────────────────────
  if (type === 'divergence') {
    const ind = asStr(c.indicator) || asStr(c.name) || 'price';
    const side = sub === 'bearish' || dir === 'bearish' ? 'bearish' : 'bullish';
    return `${side.charAt(0).toUpperCase() + side.slice(1)} divergence on ${indicatorLabel(ind)}${tfSuffix(tf)}`;
  }

  // ── Funding rate (operator + value, perp futures) ───────────────────────
  if (type === 'funding_rate') {
    if (val != null) return `Funding rate ${opPhrase(op)} ${val}% (perp futures)`;
    return `Extreme funding-rate reading`;
  }

  // ── Open interest (canonical: change_pct + operator + condition) ────────
  if (type === 'open_interest') {
    if (sub === 'rising') return 'Open interest is climbing (fresh money flowing in)';
    if (sub === 'falling') return 'Open interest is dropping (positions unwinding)';
    const changePct = asNum(c.change_pct) ?? val;
    if (changePct != null) return `Open interest ${opPhrase(op)} ${changePct}% change`;
    return 'Open-interest signal triggers';
  }

  // ── Session filter (`sessions` array) ───────────────────────────────────
  if (type === 'session') {
    const sessions = asArr(c.sessions);
    if (sessions && sessions.length > 0) {
      const labels = sessions.map((s) => String(s).replace(/_/g, ' '));
      return `Only active during the ${labels.join(', ')} session${labels.length > 1 ? 's' : ''}`;
    }
    return 'Time-of-day session filter';
  }

  // ── Price relative (reference + condition above|below|near + value) ─────
  if (type === 'price_relative') {
    const reference = asStr(c.reference) || 'daily_open';
    const refLabel = referenceLabel(reference);
    if (sub === 'above') return `Price is above ${refLabel}${tfSuffix(tf)}`;
    if (sub === 'below') return `Price is below ${refLabel}${tfSuffix(tf)}`;
    if (sub === 'near') {
      const thresh = asNum(c.threshold_pct);
      return `Price is within ${thresh ?? 2}% of ${refLabel}${tfSuffix(tf)}`;
    }
    if (val != null) return `Price ${opPhrase(op)} ${val}% from ${refLabel}${tfSuffix(tf)}`;
    return `Price compared to ${refLabel}${tfSuffix(tf)}`;
  }

  // ── Sentiment (operator + value, 0-100 score from CryptoNews) ───────────
  if (type === 'sentiment') {
    if (val != null) return `News sentiment score ${opPhrase(op)} ${val}/100`;
    return 'Bullish news sentiment shift';
  }

  // ── Liquidation cluster (direction below=long-liq, above=short-liq) ─────
  if (type === 'liquidation') {
    const liqDir = dir.toLowerCase();
    if (liqDir === 'below') return `Price approaches a long-liquidation cluster (cascade risk)${tfSuffix(tf)}`;
    if (liqDir === 'above') return `Price approaches a short-liquidation cluster (squeeze risk)${tfSuffix(tf)}`;
    return `Price approaches a liquidation cluster${tfSuffix(tf)}`;
  }

  // ── Trend reversal (condition or direction = bullish|bearish) ───────────
  if (type === 'trend_reversal') {
    const side = (sub || dir).toLowerCase();
    if (side === 'bullish') return `Bullish trend-reversal signal — downtrend losing steam${tfSuffix(tf)}`;
    if (side === 'bearish') return `Bearish trend-reversal signal — uptrend losing steam${tfSuffix(tf)}`;
    return `Trend reversal signal${tfSuffix(tf)}`;
  }

  // ── Sustained trend (trend_dir = pump|dump, periods, min_total_pct) ─────
  if (type === 'sustained_trend') {
    const trendDir = (asStr(c.trend_dir) || dir || '').toLowerCase();
    const periods = asNum(c.periods);
    const minPct = asNum(c.min_total_pct);
    const word = trendDir === 'pump' ? 'pumping' : trendDir === 'dump' ? 'dumping' : 'trending';
    const periodPart = periods != null ? ` for ${periods}+ ${tf || 'periods'}` : tfSuffix(tf);
    const pctPart = minPct != null ? ` (≥ ${minPct}% total move)` : '';
    return `Coin has been ${word}${periodPart}${pctPart}`;
  }

  // ── Generic numeric fallback ────────────────────────────────────────────
  if (type && val != null) {
    return `${humanise(type).replace(/^./, (m) => m.toUpperCase())} ${opPhrase(op)} ${val}${tfSuffix(tf)}`;
  }

  // Final fallback — name the type instead of the unhelpful "Custom rule"
  if (type) return `Custom ${humanise(type)} rule${withDir(dir)}${tfSuffix(tf)}`;
  return 'Custom entry rule';
}

function describeUniverse(cfg: Cfg): { label: string; hint?: string } {
  const universe = asObj(cfg.universe);
  if (!universe) {
    // Legacy: top-level "symbols" array
    const symbols = asArr(cfg.symbols);
    if (symbols && symbols.length > 0) {
      return { label: symbols.map(String).join(', ') };
    }
    return { label: 'All coins', hint: 'Scans every supported pair' };
  }
  const type = asStr(universe.type);
  if (type === 'specific') {
    const symbols = asArr(universe.symbols);
    if (symbols && symbols.length > 0) {
      const list = symbols.map(String);
      const display = list.length <= 4 ? list.join(', ') : `${list.slice(0, 4).join(', ')} +${list.length - 4} more`;
      return { label: display };
    }
  }
  return { label: 'All coins', hint: 'Scans every supported pair' };
}

export function StrategyConfigCard({ config }: { config?: Cfg | null }) {
  const cfg = config || {};

  // ─── Direction ───────────────────────────────────────────────────────────
  const direction = (asStr(cfg.direction) || 'LONG').toUpperCase();
  const dirColor =
    direction === 'LONG' ? colors.positive :
    direction === 'SHORT' ? colors.negative :
    colors.accent;
  const dirIcon = direction === 'LONG' ? 'trending-up' : direction === 'SHORT' ? 'trending-down' : 'swap-horizontal';

  // ─── Universe + timeframe ────────────────────────────────────────────────
  const universe = describeUniverse(cfg);
  const timeframe = asStr(cfg._timeframe) || asStr(cfg.timeframe) || null;

  // ─── Entry conditions ────────────────────────────────────────────────────
  const entryBlock = asObj(cfg.entry_conditions);
  const entryConds = entryBlock ? asArr(entryBlock.conditions) || [] : [];
  const entryOp = (asStr(entryBlock?.operator) || 'AND').toUpperCase();
  const conditionLines = entryConds
    .map((c) => asObj(c))
    .filter((c): c is Cfg => !!c)
    .map(describeCondition);

  // ─── Exit ────────────────────────────────────────────────────────────────
  const exit = asObj(cfg.exit) || {};
  const tp = asNum(exit.take_profit_pct);
  const tp2 = asNum(exit.take_profit2_pct);
  const sl = asNum(exit.stop_loss_pct);
  const trailing = exit.trailing_stop === true;
  const trailingPct = asNum(exit.trailing_stop_pct);
  const breakeven = asNum(exit.breakeven_at_pct);

  // ─── Risk ────────────────────────────────────────────────────────────────
  const risk = asObj(cfg.risk) || {};
  const lev = asNum(risk.leverage);
  const sizePct = asNum(risk.position_size_pct);
  const maxPerDay = asNum(risk.max_trades_per_day);
  const maxOpen = asNum(risk.max_open_positions);
  const cooldown = asNum(risk.cooldown_minutes);
  const riskProfile = asStr(risk.risk_profile);

  return (
    <InfoCard label="How this strategy works" icon="settings-outline" iconColor={colors.accent}>
      {/* Direction + universe summary */}
      <View style={styles.summary}>
        <View style={[styles.badge, { backgroundColor: `${dirColor}1f`, borderColor: `${dirColor}55` }]}>
          <Ionicons name={dirIcon as any} size={14} color={dirColor} />
          <Text style={[styles.badgeText, { color: dirColor }]}>{direction}</Text>
        </View>
        <View style={styles.summaryMain}>
          <Text style={styles.summaryTitle}>Watches {universe.label}</Text>
          <Text style={styles.summarySub}>
            {timeframe ? `${timeframe} timeframe` : 'Multi-timeframe'}
            {universe.hint ? ` · ${universe.hint}` : ''}
          </Text>
        </View>
      </View>

      {/* Triggers */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Triggers when</Text>
        {conditionLines.length === 0 ? (
          <InfoBullet
            text="Custom conditions configured on the web portal. Open the portal to view or edit the full setup."
            icon="information-circle-outline"
            iconColor={colors.textDim}
          />
        ) : (
          <>
            {conditionLines.map((line, i) => (
              <InfoBullet key={`c-${i}`} text={line} icon="flash-outline" iconColor={colors.accent} />
            ))}
            {conditionLines.length > 1 ? (
              <Text style={styles.opNote}>
                {entryOp === 'OR' ? 'Any one of the above can trigger an entry.' : 'All of the above must be true at the same time.'}
              </Text>
            ) : null}
          </>
        )}
      </View>

      {/* Exits */}
      <View style={styles.section}>
        <Text style={styles.sectionLabel}>Exits when</Text>
        {tp != null ? (
          <InfoBullet
            text={`Price moves ${direction === 'SHORT' ? 'down' : 'up'} ${tp}% — take profit${tp2 != null ? ` (then a second target at ${tp2}%)` : ''}`}
            icon="checkmark-circle-outline"
            iconColor={colors.positive}
          />
        ) : null}
        {sl != null ? (
          <InfoBullet
            text={`Price moves against you ${sl}% — stop loss kicks in to protect capital`}
            icon="close-circle-outline"
            iconColor={colors.negative}
          />
        ) : null}
        {trailing ? (
          <InfoBullet
            text={`Trailing stop follows price${trailingPct != null ? ` at ${trailingPct}%` : ''}, locking in gains as price moves your way`}
            icon="git-commit-outline"
            iconColor={colors.violet}
          />
        ) : null}
        {breakeven != null ? (
          <InfoBullet
            text={`Stop moves to break-even once you're up ${breakeven}%`}
            icon="shield-checkmark-outline"
            iconColor={colors.violet}
          />
        ) : null}
        {tp == null && sl == null && !trailing ? (
          <InfoBullet
            text="No automatic exit configured — positions stay open until you close them manually."
            icon="information-circle-outline"
            iconColor={colors.warning}
          />
        ) : null}
      </View>

      {/* Risk */}
      <View style={[styles.section, styles.sectionLast]}>
        <Text style={styles.sectionLabel}>Risk profile</Text>
        <View style={styles.chipRow}>
          {lev != null ? <Chip label={`${lev}× leverage`} tone="accent" /> : null}
          {sizePct != null ? <Chip label={`${sizePct}% per trade`} /> : null}
          {maxPerDay != null ? <Chip label={`max ${maxPerDay}/day`} /> : null}
          {maxOpen != null ? <Chip label={`${maxOpen} open at once`} /> : null}
          {cooldown != null ? <Chip label={`${cooldown}m cooldown`} /> : null}
          {riskProfile ? <Chip label={riskProfile} tone="violet" /> : null}
          {lev == null && sizePct == null && !riskProfile ? (
            <Text style={styles.emptyHint}>Default risk settings apply.</Text>
          ) : null}
        </View>
      </View>
    </InfoCard>
  );
}

function Chip({ label, tone = 'neutral' }: { label: string; tone?: 'neutral' | 'accent' | 'violet' }) {
  const fg = tone === 'accent' ? colors.accent : tone === 'violet' ? colors.violet : colors.text;
  const bg = tone === 'accent' ? 'rgba(255,255,255,0.10)' : tone === 'violet' ? 'rgba(255,255,255,0.10)' : 'rgba(255,255,255,0.05)';
  const bd = tone === 'accent' ? 'rgba(255,255,255,0.10)' : tone === 'violet' ? 'rgba(255,255,255,0.10)' : colors.border;
  return (
    <View style={[chipStyles.chip, { backgroundColor: bg, borderColor: bd }]}>
      <Text style={[chipStyles.chipText, { color: fg }]}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  summary: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    gap: spacing.md,
  },
  badge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: radius.pill,
    borderWidth: 1,
    gap: 4,
  },
  badgeText: { fontFamily: font.bold, fontSize: 11, letterSpacing: 0.4 },
  summaryMain: { flex: 1 },
  summaryTitle: { color: colors.text, fontFamily: font.semibold, fontSize: 14 },
  summarySub: { color: colors.textDim, fontFamily: font.regular, fontSize: 12, marginTop: 2 },
  section: {
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.divider,
  },
  sectionLast: { paddingBottom: spacing.md },
  sectionLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 10.5,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: 6,
  },
  opNote: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontStyle: 'italic',
    fontSize: 11.5,
    marginTop: 4,
    marginLeft: 22,
  },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  emptyHint: { color: colors.textMute, fontFamily: font.regular, fontSize: 12 },
});

const chipStyles = StyleSheet.create({
  chip: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: radius.pill,
    borderWidth: 1,
  },
  chipText: { fontFamily: font.bold, fontSize: 11, letterSpacing: 0.2 },
});
