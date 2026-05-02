import React from 'react';
import { View, ScrollView, RefreshControl, StyleSheet, Text, Platform } from 'react-native';
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
