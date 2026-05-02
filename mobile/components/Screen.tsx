import React from 'react';
import { View, ScrollView, RefreshControl, StyleSheet, Text, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { colors, spacing } from '@/constants/colors';

/**
 * Standard screen container with dark background, optional header,
 * and pull-to-refresh ScrollView. Handles safe-area insets correctly.
 */
export function Screen({
  children,
  title,
  subtitle,
  refreshing,
  onRefresh,
  scroll = true,
}: {
  children: React.ReactNode;
  title?: string;
  subtitle?: string;
  refreshing?: boolean;
  onRefresh?: () => void;
  scroll?: boolean;
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
      <Wrapper {...wrapperProps as any}>
        {(title || subtitle) ? (
          <View style={styles.header}>
            {title ? <Text style={styles.title}>{title}</Text> : null}
            {subtitle ? <Text style={styles.subtitle}>{subtitle}</Text> : null}
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
    paddingBottom: spacing.xxl + 80,
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
  title: {
    color: colors.text,
    fontSize: 28,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  subtitle: {
    color: colors.textDim,
    fontSize: 14,
    marginTop: 2,
  },
});
