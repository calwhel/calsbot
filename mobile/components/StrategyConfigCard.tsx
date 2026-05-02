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

/** Format an entry condition object into a plain-English sentence. */
function describeCondition(c: Cfg): string {
  const type = asStr(c.type) || '';
  const name = asStr(c.name) || '';
  const tf = asStr(c.timeframe) || '';
  const op = asStr(c.operator) || '';
  const val = asNum(c.value);

  const opWord =
    op === 'lt' || op === '<' ? 'drops below' :
    op === 'gt' || op === '>' ? 'climbs above' :
    op === 'lte' || op === '<=' ? 'is at or below' :
    op === 'gte' || op === '>=' ? 'is at or above' :
    op === 'eq' || op === '==' ? 'equals' :
    op === 'cross_above' ? 'crosses above' :
    op === 'cross_below' ? 'crosses below' :
    op || 'reaches';

  // Indicator-typed conditions (RSI, MACD, EMA…)
  if (type === 'indicator' && name) {
    const indicator = name.toUpperCase();
    const tfPart = tf ? ` on the ${tf} timeframe` : '';
    if (val != null) return `${indicator} ${opWord} ${val}${tfPart}`;
    return `${indicator} signal triggers${tfPart}`;
  }

  // Price level conditions
  if (type === 'price' && val != null) {
    return `Price ${opWord} ${val}${tf ? ` on the ${tf} timeframe` : ''}`;
  }

  // Volume / change / generic numeric
  if (type && val != null) {
    const niceType = type.replace(/_/g, ' ');
    return `${niceType.charAt(0).toUpperCase() + niceType.slice(1)} ${opWord} ${val}${tf ? ` on ${tf}` : ''}`;
  }

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
  const bg = tone === 'accent' ? 'rgba(34,211,238,0.12)' : tone === 'violet' ? 'rgba(167,139,250,0.14)' : 'rgba(255,255,255,0.05)';
  const bd = tone === 'accent' ? 'rgba(34,211,238,0.28)' : tone === 'violet' ? 'rgba(167,139,250,0.34)' : colors.border;
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
