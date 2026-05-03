import React from 'react';
import { View, RefreshControl, StyleSheet, Text, Platform } from 'react-native';
// IMPORTANT: we use react-native-gesture-handler's ScrollView (not RN's
// built-in one) so that child gestures wrapped in <GestureDetector> can
// reliably arbitrate against the parent scroll.
import { ScrollView } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, font, spacing } from '@/constants/colors';

/**
 * Screen (modern-dark) — flat background, refined header type, pull-to-refresh.
 * AmbientBg is removed entirely; the bg is `colors.bg` only.
 */
export function Screen({
  children,
  title,
  subtitle,
  rightSlot,
  refreshing,
  onRefresh,
  scroll = true,
  ambient: _ambient = 'duo',
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
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
            tintColor={colors.textDim}
            colors={[colors.text]}
            progressBackgroundColor={colors.bgElev}
          />
        ) : undefined,
        showsVerticalScrollIndicator: false,
        automaticallyAdjustContentInsets: false,
        contentInsetAdjustmentBehavior: 'never' as const,
        keyboardShouldPersistTaps: 'handled' as const,
        scrollEventThrottle: 16,
      }
    : { style: [styles.fixed, { paddingTop: topPad }] };

  return (
    <View style={styles.root}>
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
    paddingTop: spacing.lg,
    paddingBottom: spacing.xl,
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
  },
  title: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 32,
    letterSpacing: -0.8,
    lineHeight: 38,
  },
  subtitle: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    marginTop: 6,
    lineHeight: 20,
  },
});
