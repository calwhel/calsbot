import React from 'react';
import { View, RefreshControl, StyleSheet, Text, Platform } from 'react-native';
// IMPORTANT: we use react-native-gesture-handler's ScrollView (not RN's
// built-in one) so that child gestures wrapped in <GestureDetector> can
// reliably arbitrate against the parent scroll. With RN's ScrollView the
// child Pan gesture's `failOffsetY` / `activeOffsetX` are evaluated by the
// gesture-handler system but iOS's underlying UIScrollView doesn't know to
// participate, so horizontal-pan-on-chart was being intermittently swallowed.
// RNGH's ScrollView is a drop-in replacement that wires the native scroll
// into the same arbitrator the chart uses.
import { ScrollView } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { AmbientBg } from '@/components/AmbientBg';
import { colors, font, spacing } from '@/constants/colors';

/**
 * Standard screen container with the ambient gradient backdrop, optional
 * header, and pull-to-refresh ScrollView. Handles safe-area insets correctly.
 *
 * The AmbientBg is fixed (won't scroll) and absolutely positioned so the
 * blob colours stay anchored to the top of the viewport while content scrolls
 * naturally underneath.
 */
export function Screen({
  children,
  title,
  subtitle,
  rightSlot,
  refreshing,
  onRefresh,
  scroll = true,
  ambient = 'duo',
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  /** Optional element rendered to the right of the title row (e.g. button). */
  rightSlot?: React.ReactNode;
  refreshing?: boolean;
  onRefresh?: () => void;
  scroll?: boolean;
  ambient?: 'duo' | 'cyan' | 'violet' | 'none';
}) {
  const insets = useSafeAreaInsets();
  const topPad = Platform.OS === 'web' ? Math.max(insets.top, 16) : insets.top;
  const Wrapper = scroll ? ScrollView : View;

  const wrapperProps = scroll
    ? {
        style: [styles.scroll, { paddingTop: topPad }],
        contentContainerStyle: styles.scrollContent,
        refreshControl: onRefresh ? (
          <RefreshControl
            refreshing={!!refreshing}
            onRefresh={onRefresh}
            tintColor={colors.accent}
            colors={[colors.accent]}
            progressBackgroundColor={colors.bgElev}
          />
        ) : undefined,
        showsVerticalScrollIndicator: false,
        // ── iOS scroll-jump fix ─────────────────────────────────────────
        // Without these, iOS will auto-adjust contentInset whenever the
        // status-bar / keyboard / RefreshControl inset wobbles. With our
        // 1Hz ticker polling causing frequent re-renders, that auto-adjust
        // surfaces as the page "jolting back to the top" mid-scroll. We
        // own the safe-area padding via `topPad` and the bottom padding via
        // `scrollContent.paddingBottom`, so we don't need iOS doing it.
        automaticallyAdjustContentInsets: false,
        contentInsetAdjustmentBehavior: 'never' as const,
        keyboardShouldPersistTaps: 'handled' as const,
        // Cap the per-frame scroll deceleration so the chart's horizontal
        // pan can grab a touch even mid-scroll without the inertial scroll
        // fighting it.
        scrollEventThrottle: 16,
      }
    : { style: [styles.fixed, { paddingTop: topPad }] };

  return (
    <View style={styles.root}>
      <AmbientBg variant={ambient} />
      <Wrapper {...wrapperProps as any}>
        {(title || subtitle) ? (
          <View style={styles.header}>
            <View style={styles.headerRow}>
              <View style={{ flex: 1 }}>
                {title ? <Text style={styles.title}>{title}</Text> : null}
                {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
              </View>
              {rightSlot ? <View style={{ marginLeft: spacing.md }}>{rightSlot}</View> : null}
            </View>
          </View>
        ) : null}
        {children}
      </Wrapper>
    </View>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  scroll: { flex: 1 },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xxl + 96,
  },
  fixed: {
    flex: 1,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.lg,
  },
  header: {
    paddingTop: spacing.md,
    paddingBottom: spacing.lg,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  title: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 30,
    letterSpacing: -0.8,
    lineHeight: 36,
  },
  subtitle: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    marginTop: 4,
    lineHeight: 19,
  },
});
