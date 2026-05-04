import React, { useMemo, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  Pressable,
  ScrollView,
  ActivityIndicator,
  Switch,
  Platform,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';

import { colors, font, radius, spacing } from '@/constants/colors';
import { postQuickTrade, type QuickTradeResult } from '@/lib/api';
import { useAuth } from '@/contexts/AuthContext';
import { RiskDisclaimer } from '@/components/RiskDisclaimer';

type Props = {
  visible: boolean;
  onClose: () => void;
  symbol: string;          // current chart symbol e.g. "BTC"
  livePrice: number | null;
};

const LEV_PRESETS = [3, 5, 10, 20, 50];
const SIZE_PRESETS = [10, 25, 50, 100, 250];   // USD margin
const TP_PRESETS = [2, 5, 10, 20];             // percent
const SL_PRESETS = [1, 2, 5, 10];

function fmtPrice(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '—';
  if (p >= 1000) return p.toLocaleString(undefined, { maximumFractionDigits: 2 });
  if (p >= 1)    return p.toLocaleString(undefined, { maximumFractionDigits: 3 });
  if (p >= 0.01) return p.toLocaleString(undefined, { maximumFractionDigits: 5 });
  return p.toLocaleString(undefined, { maximumFractionDigits: 8 });
}

export function QuickTradeSheet({ visible, onClose, symbol, livePrice }: Props) {
  const { uid } = useAuth();
  const [side, setSide] = useState<'LONG' | 'SHORT'>('LONG');
  const [leverage, setLeverage] = useState<number>(10);
  const [sizeUsd, setSizeUsd] = useState<number>(25);
  const [useTp, setUseTp] = useState<boolean>(true);
  const [tpPct, setTpPct] = useState<number>(5);
  const [useSl, setUseSl] = useState<boolean>(true);
  const [slPct, setSlPct] = useState<number>(2);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [result, setResult] = useState<QuickTradeResult | null>(null);
  const [errMsg, setErrMsg] = useState<string | null>(null);

  // Resulting price preview for TP/SL
  const tpPrice = useMemo(() => {
    if (!livePrice || !useTp) return null;
    return side === 'LONG' ? livePrice * (1 + tpPct / 100) : livePrice * (1 - tpPct / 100);
  }, [livePrice, useTp, tpPct, side]);
  const slPrice = useMemo(() => {
    if (!livePrice || !useSl) return null;
    return side === 'LONG' ? livePrice * (1 - slPct / 100) : livePrice * (1 + slPct / 100);
  }, [livePrice, useSl, slPct, side]);

  const notional = sizeUsd * leverage;
  const sideColor = side === 'LONG' ? colors.positive : colors.negative;

  function reset() {
    setResult(null);
    setErrMsg(null);
  }

  function close() {
    if (submitting) return;
    reset();
    onClose();
  }

  async function fire() {
    if (submitting || !uid) return;
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(() => {});
    setSubmitting(true);
    setResult(null);
    setErrMsg(null);
    try {
      const res = await postQuickTrade(uid, {
        symbol,
        side,
        leverage,
        position_usd: sizeUsd,
        tp_pct: useTp ? tpPct : null,
        sl_pct: useSl ? slPct : null,
      });
      setResult(res);
      if (res.ok) {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      } else {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
        setErrMsg(res.message || res.error || 'Order rejected');
      }
    } catch (e: any) {
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error).catch(() => {});
      setErrMsg(e?.message || String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={close}
      statusBarTranslucent
    >
      <Pressable style={styles.backdrop} onPress={close}>
        <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
          {/* Drag handle + header */}
          <View style={styles.handleWrap}>
            <View style={styles.handle} />
          </View>
          <View style={styles.headerRow}>
            <Text style={styles.title}>Quick Trade</Text>
            <Pressable onPress={close} hitSlop={12} style={styles.closeBtn}>
              <Ionicons name="close" size={22} color={colors.textDim} />
            </Pressable>
          </View>

          <ScrollView style={{ maxHeight: 560 }} contentContainerStyle={{ paddingBottom: spacing.lg }}>
            {/* Symbol + price banner */}
            <View style={styles.symBanner}>
              <Text style={styles.symBig}>{symbol}/USDT</Text>
              <Text style={styles.symPrice}>${fmtPrice(livePrice)}</Text>
            </View>

            {/* Side toggle */}
            <View style={styles.sideRow}>
              <Pressable
                onPress={() => { Haptics.selectionAsync().catch(() => {}); setSide('LONG'); reset(); }}
                style={[
                  styles.sideBtn,
                  side === 'LONG' && { backgroundColor: colors.positive, borderColor: colors.positive },
                ]}
              >
                <Ionicons name="trending-up" size={18} color={side === 'LONG' ? '#0b1410' : colors.positive} />
                <Text style={[styles.sideBtnText, side === 'LONG' && { color: '#0b1410' }]}>LONG</Text>
              </Pressable>
              <Pressable
                onPress={() => { Haptics.selectionAsync().catch(() => {}); setSide('SHORT'); reset(); }}
                style={[
                  styles.sideBtn,
                  side === 'SHORT' && { backgroundColor: colors.negative, borderColor: colors.negative },
                ]}
              >
                <Ionicons name="trending-down" size={18} color={side === 'SHORT' ? '#1a0c0c' : colors.negative} />
                <Text style={[styles.sideBtnText, side === 'SHORT' && { color: '#1a0c0c' }]}>SHORT</Text>
              </Pressable>
            </View>

            {/* Leverage */}
            <Text style={styles.label}>Leverage</Text>
            <View style={styles.chipRow}>
              {LEV_PRESETS.map((lv) => (
                <Pressable
                  key={lv}
                  onPress={() => { Haptics.selectionAsync().catch(() => {}); setLeverage(lv); }}
                  style={[styles.chip, leverage === lv && styles.chipActive]}
                >
                  <Text style={[styles.chipText, leverage === lv && styles.chipTextActive]}>{lv}×</Text>
                </Pressable>
              ))}
            </View>

            {/* Margin (USD) */}
            <Text style={styles.label}>Margin <Text style={styles.labelSub}>(USDT)</Text></Text>
            <View style={styles.chipRow}>
              {SIZE_PRESETS.map((s) => (
                <Pressable
                  key={s}
                  onPress={() => { Haptics.selectionAsync().catch(() => {}); setSizeUsd(s); }}
                  style={[styles.chip, sizeUsd === s && styles.chipActive]}
                >
                  <Text style={[styles.chipText, sizeUsd === s && styles.chipTextActive]}>${s}</Text>
                </Pressable>
              ))}
            </View>
            <Text style={styles.notional}>
              Notional: <Text style={{ color: colors.text }}>${notional.toLocaleString()}</Text>
            </Text>

            {/* TP */}
            <View style={styles.toggleRow}>
              <Text style={styles.label}>Take Profit</Text>
              <Switch
                value={useTp}
                onValueChange={setUseTp}
                trackColor={{ false: '#333', true: colors.accent }}
                thumbColor={Platform.OS === 'android' ? colors.text : undefined}
              />
            </View>
            {useTp ? (
              <>
                <View style={styles.chipRow}>
                  {TP_PRESETS.map((p) => (
                    <Pressable
                      key={p}
                      onPress={() => { Haptics.selectionAsync().catch(() => {}); setTpPct(p); }}
                      style={[styles.chip, tpPct === p && styles.chipActive]}
                    >
                      <Text style={[styles.chipText, tpPct === p && styles.chipTextActive]}>+{p}%</Text>
                    </Pressable>
                  ))}
                </View>
                <Text style={styles.preview}>
                  Target: <Text style={{ color: colors.positive }}>${fmtPrice(tpPrice)}</Text>
                </Text>
              </>
            ) : null}

            {/* SL */}
            <View style={styles.toggleRow}>
              <Text style={styles.label}>Stop Loss</Text>
              <Switch
                value={useSl}
                onValueChange={setUseSl}
                trackColor={{ false: '#333', true: colors.accent }}
                thumbColor={Platform.OS === 'android' ? colors.text : undefined}
              />
            </View>
            {useSl ? (
              <>
                <View style={styles.chipRow}>
                  {SL_PRESETS.map((p) => (
                    <Pressable
                      key={p}
                      onPress={() => { Haptics.selectionAsync().catch(() => {}); setSlPct(p); }}
                      style={[styles.chip, slPct === p && styles.chipActive]}
                    >
                      <Text style={[styles.chipText, slPct === p && styles.chipTextActive]}>−{p}%</Text>
                    </Pressable>
                  ))}
                </View>
                <Text style={styles.preview}>
                  Stop: <Text style={{ color: colors.negative }}>${fmtPrice(slPrice)}</Text>
                </Text>
              </>
            ) : null}

            {/* Result / error */}
            {result?.ok ? (
              <View style={[styles.banner, { borderColor: colors.positive, backgroundColor: 'rgba(34,197,94,0.08)' }]}>
                <Ionicons name="checkmark-circle" size={18} color={colors.positive} />
                <View style={{ flex: 1, marginLeft: 8 }}>
                  <Text style={[styles.bannerTitle, { color: colors.positive }]}>Order placed</Text>
                  <Text style={styles.bannerBody}>
                    {result.symbol} {result.side} {result.leverage}× @ ${fmtPrice(result.entry_price ?? null)}
                    {result.order_id ? `\nOrder ID: ${result.order_id}` : ''}
                  </Text>
                </View>
              </View>
            ) : null}
            {errMsg ? (
              <View style={[styles.banner, { borderColor: colors.negative, backgroundColor: 'rgba(248,113,113,0.08)' }]}>
                <Ionicons name="alert-circle" size={18} color={colors.negative} />
                <View style={{ flex: 1, marginLeft: 8 }}>
                  <Text style={[styles.bannerTitle, { color: colors.negative }]}>Could not place</Text>
                  <Text style={styles.bannerBody}>{errMsg}</Text>
                </View>
              </View>
            ) : null}
          </ScrollView>

          <RiskDisclaimer compact />

          {/* Confirm */}
          <Pressable
            onPress={fire}
            disabled={submitting || !livePrice || !uid}
            style={({ pressed }) => [
              styles.confirmBtn,
              { backgroundColor: sideColor },
              (pressed || submitting) && { opacity: 0.85 },
              (!livePrice || !uid) && { opacity: 0.45 },
            ]}
          >
            {submitting ? (
              <ActivityIndicator color="#000" />
            ) : (
              <Text style={[styles.confirmText, { color: side === 'LONG' ? '#0b1410' : '#1a0c0c' }]}>
                {side === 'LONG' ? 'Buy' : 'Sell'} {symbol} · {leverage}× · ${sizeUsd}
              </Text>
            )}
          </Pressable>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.55)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: colors.bg,
    borderTopLeftRadius: 24,
    borderTopRightRadius: 24,
    paddingHorizontal: spacing.md,
    paddingTop: 6,
    paddingBottom: spacing.lg,
    borderTopWidth: 1,
    borderColor: colors.border,
  },
  handleWrap: { alignItems: 'center', paddingVertical: 8 },
  handle: { width: 40, height: 4, borderRadius: 2, backgroundColor: '#444' },
  headerRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingBottom: spacing.sm,
  },
  title: { color: colors.text, fontFamily: font.bold, fontSize: 18 },
  closeBtn: { padding: 4 },
  symBanner: {
    flexDirection: 'row', alignItems: 'baseline', justifyContent: 'space-between',
    paddingVertical: 10, paddingHorizontal: 12,
    backgroundColor: colors.card, borderRadius: radius.lg,
    borderWidth: 1, borderColor: colors.border,
    marginBottom: spacing.md,
  },
  symBig: { color: colors.text, fontFamily: font.bold, fontSize: 16 },
  symPrice: {
    color: colors.text, fontFamily: font.bold, fontSize: 18,
    fontVariant: ['tabular-nums'],
  },
  sideRow: { flexDirection: 'row', gap: spacing.sm, marginBottom: spacing.md },
  sideBtn: {
    flex: 1, flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    gap: 8, paddingVertical: 14, borderRadius: radius.lg,
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border,
  },
  sideBtnText: { color: colors.text, fontFamily: font.bold, fontSize: 14, letterSpacing: 0.5 },
  label: {
    color: colors.textDim, fontFamily: font.semibold, fontSize: 12,
    textTransform: 'uppercase', letterSpacing: 0.6,
    marginTop: spacing.md, marginBottom: 8,
  },
  labelSub: { color: colors.textMute, fontFamily: font.regular },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  chip: {
    paddingHorizontal: 14, paddingVertical: 8, borderRadius: 999,
    backgroundColor: colors.card, borderWidth: 1, borderColor: colors.border,
  },
  chipActive: { backgroundColor: colors.accent, borderColor: colors.accent },
  chipText: { color: colors.text, fontFamily: font.semibold, fontSize: 13 },
  chipTextActive: { color: colors.accentText },
  notional: { color: colors.textMute, fontSize: 12, fontFamily: font.regular, marginTop: 8 },
  toggleRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    marginTop: spacing.md,
  },
  preview: {
    color: colors.textDim, fontFamily: font.semibold, fontSize: 13,
    marginTop: 8, fontVariant: ['tabular-nums'],
  },
  banner: {
    flexDirection: 'row', alignItems: 'flex-start',
    padding: 12, borderRadius: radius.md, borderWidth: 1,
    marginTop: spacing.md,
  },
  bannerTitle: { fontFamily: font.bold, fontSize: 13, marginBottom: 2 },
  bannerBody: { color: colors.text, fontFamily: font.regular, fontSize: 12.5, lineHeight: 18 },
  confirmBtn: {
    marginTop: spacing.md, paddingVertical: 16, borderRadius: radius.lg,
    alignItems: 'center', justifyContent: 'center',
  },
  confirmText: { fontFamily: font.bold, fontSize: 15, letterSpacing: 0.4 },
});
