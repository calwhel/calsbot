import React, { useEffect, useRef } from 'react';
import { Animated, View, StyleSheet, Easing, ViewStyle, StyleProp } from 'react-native';
import { colors, radius, spacing } from '@/constants/colors';

/**
 * Skeleton — single animated shimmer block. Use directly with `width`/`height`
 * for arbitrary placeholder shapes, or use the preset components below for
 * common card layouts.
 */
export function Skeleton({
  width,
  height = 12,
  radius: r = 6,
  style,
}: {
  width?: number | `${number}%`;
  height?: number;
  radius?: number;
  style?: StyleProp<ViewStyle>;
}) {
  const opacity = useRef(new Animated.Value(0.5)).current;

  useEffect(() => {
    const loop = Animated.loop(
      Animated.sequence([
        Animated.timing(opacity, {
          toValue: 1,
          duration: 850,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
        Animated.timing(opacity, {
          toValue: 0.5,
          duration: 850,
          easing: Easing.inOut(Easing.ease),
          useNativeDriver: true,
        }),
      ]),
    );
    loop.start();
    return () => loop.stop();
  }, [opacity]);

  return (
    <Animated.View
      style={[
        {
          width: (width as any) ?? '100%',
          height,
          borderRadius: r,
          backgroundColor: colors.cardHi,
          opacity,
        },
        style,
      ]}
    />
  );
}

/** Hero P&L card placeholder — matches MeshHero footprint. */
export function HeroSkeleton() {
  return (
    <View style={styles.heroCard}>
      <Skeleton width={80} height={10} />
      <View style={{ height: spacing.md }} />
      <Skeleton width={'60%'} height={48} radius={8} />
      <View style={{ height: spacing.md }} />
      <Skeleton width={'85%'} height={12} />
      <View style={{ height: spacing.lg }} />
      <Skeleton width={'100%'} height={44} radius={4} />
    </View>
  );
}

/** Strategy card placeholder. */
export function StrategyCardSkeleton() {
  return (
    <View style={styles.card}>
      <View style={styles.row}>
        <Skeleton width={36} height={36} radius={10} />
        <View style={{ flex: 1, marginLeft: spacing.md }}>
          <Skeleton width={'70%'} height={14} />
          <View style={{ height: 8 }} />
          <Skeleton width={'40%'} height={11} />
        </View>
        <Skeleton width={56} height={20} radius={999} />
      </View>
      <View style={{ height: spacing.lg }} />
      <View style={styles.row}>
        <Skeleton width={'30%'} height={26} radius={6} />
        <View style={{ width: spacing.md }} />
        <Skeleton width={'30%'} height={26} radius={6} />
        <View style={{ width: spacing.md }} />
        <Skeleton width={'30%'} height={26} radius={6} />
      </View>
    </View>
  );
}

/** List of strategy card placeholders. */
export function StrategyListSkeleton({ count = 4 }: { count?: number }) {
  return (
    <View style={{ gap: spacing.md }}>
      {Array.from({ length: count }).map((_, i) => (
        <StrategyCardSkeleton key={i} />
      ))}
    </View>
  );
}

/** Marketplace listing card placeholder. */
export function MarketplaceCardSkeleton() {
  return (
    <View style={styles.card}>
      <View style={styles.row}>
        <Skeleton width={44} height={44} radius={22} />
        <View style={{ flex: 1, marginLeft: spacing.md }}>
          <Skeleton width={'80%'} height={15} />
          <View style={{ height: 8 }} />
          <Skeleton width={'50%'} height={11} />
        </View>
      </View>
      <View style={{ height: spacing.lg }} />
      <Skeleton width={'100%'} height={56} radius={8} />
      <View style={{ height: spacing.md }} />
      <View style={styles.row}>
        <Skeleton width={64} height={20} radius={999} />
        <View style={{ width: 8 }} />
        <Skeleton width={56} height={20} radius={999} />
        <View style={{ flex: 1 }} />
        <Skeleton width={70} height={28} radius={6} />
      </View>
    </View>
  );
}

export function MarketplaceListSkeleton({ count = 3 }: { count?: number }) {
  return (
    <View style={{ gap: spacing.md }}>
      {Array.from({ length: count }).map((_, i) => (
        <MarketplaceCardSkeleton key={i} />
      ))}
    </View>
  );
}

/** Home composite skeleton — hero + small action row + tile grid. */
export function HomeSkeleton() {
  return (
    <View style={{ gap: spacing.lg }}>
      <HeroSkeleton />
      <View style={[styles.row, { gap: spacing.md }]}>
        <Skeleton width={68} height={70} radius={12} />
        <Skeleton width={68} height={70} radius={12} />
        <Skeleton width={68} height={70} radius={12} />
        <Skeleton width={68} height={70} radius={12} />
      </View>
      <View style={[styles.row, { gap: spacing.md }]}>
        <View style={{ flex: 1 }}><Skeleton height={104} radius={12} /></View>
        <View style={{ flex: 1 }}><Skeleton height={104} radius={12} /></View>
      </View>
      <StrategyListSkeleton count={2} />
    </View>
  );
}

const styles = StyleSheet.create({
  heroCard: {
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    padding: spacing.xl,
  },
  card: {
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    backgroundColor: colors.card,
    padding: spacing.lg,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
  },
});
