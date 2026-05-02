import React, { useMemo } from 'react';
import {
  Modal, View, Text, Pressable, StyleSheet, ScrollView, Platform,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import { Ionicons } from '@expo/vector-icons';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import {
  SIGNAL_META, CATEGORY_LABEL, type SignalType, type SignalCategory, type StyleId,
  STYLE_SIGNALS,
} from '@/lib/strategyPresets';
import { colors, font, radius, spacing } from '@/constants/colors';

/**
 * Bottom-sheet modal that lists every supported entry/confirmation signal
 * grouped by category, with a "Recommended for your style" section pinned at
 * the top. Used for picking the primary entry signal AND confirmations.
 */
export function ConditionPicker({
  visible,
  onClose,
  onPick,
  current,
  style,
  title = 'Pick a signal',
  excludeTypes = [],
}: {
  visible: boolean;
  onClose: () => void;
  onPick: (type: SignalType) => void;
  current: SignalType | null;
  style: StyleId | null;
  title?: string;
  /** Hide types the user has already added (e.g. existing confirmations). */
  excludeTypes?: SignalType[];
}) {
  const insets = useSafeAreaInsets();

  const grouped = useMemo(() => {
    const all = Object.entries(SIGNAL_META) as [SignalType, typeof SIGNAL_META[SignalType]][];
    const byCat: Partial<Record<SignalCategory, [SignalType, typeof SIGNAL_META[SignalType]][]>> = {};
    for (const [t, meta] of all) {
      if (excludeTypes.includes(t)) continue;
      (byCat[meta.category] ||= []).push([t, meta]);
    }
    return byCat;
  }, [excludeTypes]);

  const recommended = useMemo(() => {
    if (!style) return [];
    return STYLE_SIGNALS[style].filter(t => !excludeTypes.includes(t));
  }, [style, excludeTypes]);

  const handlePick = (t: SignalType) => {
    if (Platform.OS !== 'web') {
      Haptics.selectionAsync().catch(() => {});
    }
    onPick(t);
    onClose();
  };

  return (
    <Modal
      visible={visible}
      animationType="slide"
      transparent
      onRequestClose={onClose}
      statusBarTranslucent
    >
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable
          style={[styles.sheet, { paddingBottom: Math.max(insets.bottom, 12) + 12 }]}
          onPress={() => {}}
        >
          <View style={styles.header}>
            <Text style={styles.title}>{title}</Text>
            <Pressable onPress={onClose} hitSlop={12} accessibilityRole="button" accessibilityLabel="Close picker">
              <Ionicons name="close" size={22} color={colors.textDim} />
            </Pressable>
          </View>

          <ScrollView contentContainerStyle={{ paddingBottom: spacing.md }} showsVerticalScrollIndicator={false}>
            {recommended.length ? (
              <View style={styles.section}>
                <Text style={styles.sectionLabel}>⭐ Recommended for your style</Text>
                {recommended.map(t => (
                  <SignalRow
                    key={`rec-${t}`}
                    type={t}
                    selected={current === t}
                    onPress={() => handlePick(t)}
                    recommended
                  />
                ))}
              </View>
            ) : null}

            {(Object.entries(grouped) as [SignalCategory, [SignalType, any][]][])
              .sort(([a], [b]) => CAT_ORDER.indexOf(a) - CAT_ORDER.indexOf(b))
              .map(([cat, list]) => (
                <View key={cat} style={styles.section}>
                  <Text style={styles.sectionLabel}>{CATEGORY_LABEL[cat]}</Text>
                  {list.map(([t]) => (
                    <SignalRow
                      key={t}
                      type={t}
                      selected={current === t}
                      onPress={() => handlePick(t)}
                    />
                  ))}
                </View>
              ))}
          </ScrollView>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const CAT_ORDER: SignalCategory[] = ['oscillator', 'trend', 'price', 'volume', 'structure', 'filter'];

function SignalRow({
  type, selected, onPress, recommended = false,
}: {
  type: SignalType; selected: boolean; onPress: () => void; recommended?: boolean;
}) {
  const meta = SIGNAL_META[type];
  const freqLabel = meta.freq === 'high' ? 'Fires often' : meta.freq === 'med' ? 'Medium' : 'Rare';
  const freqColor = meta.freq === 'high' ? colors.positive : meta.freq === 'med' ? colors.warning : colors.textMute;
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.row,
        selected && styles.rowActive,
        pressed && { opacity: 0.85 },
      ]}
      accessibilityRole="button"
      accessibilityLabel={`Pick ${meta.label} signal`}
      accessibilityState={{ selected }}
    >
      <Text style={styles.rowIcon}>{meta.icon}</Text>
      <View style={{ flex: 1, minWidth: 0 }}>
        <View style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
          <Text style={[styles.rowLabel, selected && styles.rowLabelActive]} numberOfLines={1}>
            {meta.label}
          </Text>
          {recommended ? <Text style={styles.recBadge}>★</Text> : null}
        </View>
        <Text style={styles.rowDesc} numberOfLines={1}>{meta.desc}</Text>
      </View>
      <View style={[styles.freqDot, { backgroundColor: freqColor }]} />
      <Text style={[styles.freqLabel, { color: freqColor }]}>{freqLabel}</Text>
      {selected ? <Ionicons name="checkmark-circle" size={18} color={colors.accent} style={{ marginLeft: 4 }} /> : null}
    </Pressable>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.55)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: colors.bgElev,
    borderTopLeftRadius: radius.xl,
    borderTopRightRadius: radius.xl,
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    maxHeight: '85%',
    borderTopWidth: 1, borderColor: colors.border,
  },
  header: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingBottom: spacing.md,
    borderBottomWidth: 1, borderBottomColor: colors.border,
    marginBottom: spacing.sm,
  },
  title:   { fontFamily: font.bold, fontSize: 16, color: colors.text },
  section: { marginTop: spacing.md },
  sectionLabel: {
    fontFamily: font.bold, fontSize: 11, letterSpacing: 1.2,
    color: colors.textDim, textTransform: 'uppercase',
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row', alignItems: 'center', gap: 10,
    paddingVertical: 12, paddingHorizontal: 12,
    borderRadius: radius.lg,
    backgroundColor: colors.card,
    borderWidth: 1, borderColor: colors.border,
    marginBottom: 6,
  },
  rowActive: { borderColor: colors.accent, backgroundColor: colors.cardHi },
  rowIcon:   { fontSize: 18, width: 24, textAlign: 'center' },
  rowLabel:  { fontFamily: font.semibold, fontSize: 13.5, color: colors.text },
  rowLabelActive: { color: colors.accent },
  rowDesc:   { fontFamily: font.regular, fontSize: 11.5, color: colors.textMute, marginTop: 2 },
  freqDot:   { width: 6, height: 6, borderRadius: 3 },
  freqLabel: { fontFamily: font.medium, fontSize: 10, marginLeft: 2 },
  recBadge:  { color: colors.warning, fontSize: 11 },
});
