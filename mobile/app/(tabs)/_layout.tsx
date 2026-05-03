import React from 'react';
import { Tabs } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { View, Text, Pressable, StyleSheet, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import * as Haptics from 'expo-haptics';
import type { BottomTabBarProps } from '@react-navigation/bottom-tabs';
import { colors, font } from '@/constants/colors';

const ICONS: Record<string, keyof typeof Ionicons.glyphMap> = {
  index:        'home',
  strategies:   'pulse',
  trade:        'analytics',
  trades:       'swap-horizontal',
  marketplace:  'storefront',
  settings:     'person-circle',
};

const LABELS: Record<string, string> = {
  index:       'Home',
  strategies:  'Strategies',
  trade:       'Trade',
  trades:      'History',
  marketplace: 'Market',
  settings:    'Account',
};

/**
 * Modern-dark tab bar — bottom-anchored, hairline divider, no glow, no
 * floating pill. Active state is a quiet underline + filled icon, label
 * always visible. Reads as a tool, not as a feature.
 */
function StaticTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  const bottomPad = Math.max(insets.bottom, 8);

  return (
    <View style={[styles.outer, { paddingBottom: bottomPad }]}>
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
              <Ionicons
                name={iconName}
                size={22}
                color={focused ? colors.text : colors.textMute}
              />
              <Text
                style={[
                  styles.label,
                  { color: focused ? colors.text : colors.textMute },
                  focused && styles.labelActive,
                ]}
                numberOfLines={1}
              >
                {label}
              </Text>
              <View style={[styles.dot, focused && styles.dotActive]} />
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
      tabBar={(props) => <StaticTabBar {...props} />}
    >
      <Tabs.Screen name="index" />
      <Tabs.Screen name="strategies" />
      <Tabs.Screen name="trade" />
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
    backgroundColor: colors.bg,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  bar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    paddingTop: 10,
  },
  item: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
    gap: 3,
  },
  label: {
    fontFamily: font.medium,
    fontSize: 10,
    letterSpacing: 0.3,
  },
  labelActive: {
    fontFamily: font.semibold,
  },
  dot: {
    width: 4,
    height: 4,
    borderRadius: 2,
    marginTop: 2,
    backgroundColor: 'transparent',
  },
  dotActive: {
    backgroundColor: colors.accent,
  },
});
