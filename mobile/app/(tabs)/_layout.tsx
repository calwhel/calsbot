import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { View, Text, Pressable, StyleSheet, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import Svg, { Defs, LinearGradient as SvgLinearGradient, Stop, Rect } from 'react-native-svg';
import type { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { colors, font, glow, radius } from '@/constants/colors';

const ICONS: Record<string, keyof typeof Ionicons.glyphMap> = {
  index:        'home',
  strategies:   'pulse',
  trades:       'swap-horizontal',
  marketplace:  'storefront',
  settings:     'person-circle',
};

const LABELS: Record<string, string> = {
  index:       'Home',
  strategies:  'Strategies',
  trades:      'Trades',
  marketplace: 'Market',
  settings:    'Account',
};

/**
 * Floating pill-style tab bar. Lifts off the bottom edge with a glow, has a
 * coloured pill behind the active item, and uses Inter for labels.
 */
function FloatingTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  // Honour the safe-area inset on BOTH platforms — Android gesture-nav devices
  // report `insets.bottom` (~16-24px) and a hardcoded `12` would let the
  // floating bar overlap the system gesture pill.
  const bottomPad = Math.max(insets.bottom, 12);

  return (
    <View
      style={[styles.outer, { paddingBottom: bottomPad }]}
      pointerEvents="box-none"
    >
      <View style={styles.bar}>
        {state.routes.map((route, idx) => {
          const focused = state.index === idx;
          const { options } = descriptors[route.key];
          const label = LABELS[route.name] ?? options.title ?? route.name;
          const iconName = ICONS[route.name] ?? 'ellipse';

          const onPress = () => {
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });
            if (!focused && !event.defaultPrevented) {
              if (Platform.OS !== 'web') {
                Haptics.selectionAsync().catch(() => {});
              }
              navigation.navigate(route.name as never);
            }
          };

          return (
            <Pressable
              key={route.key}
              accessibilityRole="button"
              accessibilityState={focused ? { selected: true } : {}}
              accessibilityLabel={label}
              onPress={onPress}
              style={styles.item}
              hitSlop={8}
            >
              <View style={[styles.itemInner, focused && styles.itemActive]}>
                {focused ? (
                  <Svg
                    style={styles.activeUnderline}
                    pointerEvents="none"
                  >
                    <Defs>
                      <SvgLinearGradient id={`tab-und-${idx}`} x1="0" y1="0" x2="1" y2="0">
                        <Stop offset="0" stopColor="#22d3ee" stopOpacity="0" />
                        <Stop offset="0.5" stopColor="#22d3ee" stopOpacity="0.95" />
                        <Stop offset="1" stopColor="#6366f1" stopOpacity="0" />
                      </SvgLinearGradient>
                    </Defs>
                    <Rect width="100%" height="100%" fill={`url(#tab-und-${idx})`} />
                  </Svg>
                ) : null}
                <Ionicons
                  name={iconName}
                  size={focused ? 22 : 21}
                  color={focused ? colors.accent : colors.textMute}
                />
                {focused ? (
                  <Text style={styles.label} numberOfLines={1}>
                    {label}
                  </Text>
                ) : null}
              </View>
            </Pressable>
          );
        })}
      </View>
    </View>
  );
}

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{ headerShown: false, tabBarHideOnKeyboard: true }}
      tabBar={(props) => <FloatingTabBar {...props} />}
    >
      <Tabs.Screen name="index" />
      <Tabs.Screen name="strategies" />
      <Tabs.Screen name="trades" />
      <Tabs.Screen name="marketplace" />
      <Tabs.Screen name="settings" />
    </Tabs>
  );
}

const styles = StyleSheet.create({
  outer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    paddingHorizontal: 16,
    backgroundColor: 'transparent',
  },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.glassBg,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.08)',
    paddingHorizontal: 6,
    paddingVertical: 6,
    ...glow.card,
  },
  item: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  itemInner: {
    minHeight: 44,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    paddingHorizontal: 14,
    borderRadius: radius.pill,
  },
  itemActive: {
    backgroundColor: colors.accentDim,
    borderWidth: 1,
    borderColor: colors.borderAccent,
  },
  activeUnderline: {
    position: 'absolute',
    left: 8,
    right: 8,
    bottom: -4,
    height: 2,
    borderRadius: 1,
  },
  label: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.3,
  },
});
