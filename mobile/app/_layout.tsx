import 'react-native-gesture-handler';
import { Stack, useRouter, useSegments } from 'expo-router';
import { QueryClientProvider } from '@tanstack/react-query';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StatusBar } from 'expo-status-bar';
import { useEffect } from 'react';
import { View, ActivityIndicator, StyleSheet, Platform } from 'react-native';
import * as SplashScreen from 'expo-splash-screen';
import * as Notifications from 'expo-notifications';
import {
  useFonts,
  Inter_400Regular,
  Inter_500Medium,
  Inter_600SemiBold,
  Inter_700Bold,
  Inter_800ExtraBold,
} from '@expo-google-fonts/inter';

import { AuthProvider, useAuth } from '@/contexts/AuthContext';
import { queryClient } from '@/lib/queryClient';
import { colors } from '@/constants/colors';
import { OnboardingTour } from '@/components/OnboardingTour';

SplashScreen.preventAutoHideAsync().catch(() => {});

if (Platform.OS === 'android') {
  Notifications.setNotificationChannelAsync('trade-fires', {
    name: 'Trade Opened',
    importance: Notifications.AndroidImportance.HIGH,
    sound: 'default',
    vibrationPattern: [0, 250, 250, 250],
  });
  Notifications.setNotificationChannelAsync('trade-results', {
    name: 'Trade Results (TP/SL)',
    importance: Notifications.AndroidImportance.MAX,
    sound: 'default',
    vibrationPattern: [0, 400, 200, 400],
  });
  Notifications.setNotificationChannelAsync('trade-progress', {
    name: 'Trade Progress (TP Hit, Breakeven)',
    importance: Notifications.AndroidImportance.DEFAULT,
    sound: 'default',
  });
}

function AuthGate() {
  const { ready, user } = useAuth();
  const segments = useSegments();
  const router = useRouter();

  const first = segments[0];
  const inLogin = first === 'login';
  // A redirect is needed when the auth state disagrees with the current route.
  // We must gate the Stack render until the redirect settles — otherwise the
  // initial route (e.g. (tabs)/index) mounts for a frame and fires API calls
  // with no UID before the replace() takes effect.
  const needsRedirect = ready && ((!user && !inLogin) || (user && inLogin));

  useEffect(() => {
    if (!ready) return;
    if (!user && !inLogin) {
      router.replace('/login');
    } else if (user && inLogin) {
      router.replace('/(tabs)');
    }
  }, [ready, user, inLogin, router]);

  // Tap a push notification → deep link to the strategy detail screen if the
  // payload carries a strategy_id. Also handles the "cold start" case where
  // the app was launched from a notification tap.
  useEffect(() => {
    if (!user) return;
    const open = (data: any) => {
      if (data?.screen) {
        router.push(data.screen as any);
        return;
      }
      const sid = data?.strategy_id;
      if (sid != null) {
        router.push(`/strategy/${sid}` as any);
      }
    };
    // Cold start
    Notifications.getLastNotificationResponseAsync().then((resp) => {
      if (resp?.notification?.request?.content?.data) {
        open(resp.notification.request.content.data);
      }
    }).catch(() => {});
    // Warm taps
    const sub = Notifications.addNotificationResponseReceivedListener((resp) => {
      open(resp.notification.request.content.data);
    });
    return () => sub.remove();
  }, [user, router]);

  if (!ready || needsRedirect) {
    return (
      <View style={styles.loadingScreen}>
        <ActivityIndicator color={colors.accent} size="large" />
      </View>
    );
  }

  return (
    <>
    <Stack
      screenOptions={{
        headerShown: false,
        contentStyle: { backgroundColor: colors.bg },
        animation: 'fade',
      }}
    >
      <Stack.Screen name="(tabs)" />
      <Stack.Screen name="login" />
      <Stack.Screen
        name="strategy/[id]"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: '',
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="listing/[id]"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: '',
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="backtest/[id]"
        options={{
          headerShown: false,
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="wizard/index"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'New strategy',
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="build/index"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'New strategy',
          animation: 'slide_from_bottom',
          presentation: 'modal',
        }}
      />
      <Stack.Screen
        name="build/chat"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'AI Chat Builder',
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="build/indicator"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'AI Indicator',
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="build/pine"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'Pine Script',
          animation: 'slide_from_right',
        }}
      />
      <Stack.Screen
        name="build/scanner"
        options={{
          headerShown: true,
          headerStyle: { backgroundColor: colors.bg },
          headerTintColor: colors.text,
          headerTitleStyle: { color: colors.text, fontWeight: '700' },
          headerShadowVisible: false,
          title: 'Best Scanner',
          animation: 'slide_from_right',
        }}
      />
    </Stack>
    {/* First-launch tour — only mounts when the user is signed in so we
        don't pop a tour over the login screen. Self-hides after first run. */}
    {user ? <OnboardingTour /> : null}
    </>
  );
}

export default function RootLayout() {
  const [fontsLoaded, fontError] = useFonts({
    Inter_400Regular,
    Inter_500Medium,
    Inter_600SemiBold,
    Inter_700Bold,
    Inter_800ExtraBold,
  });

  useEffect(() => {
    if (fontsLoaded || fontError) {
      SplashScreen.hideAsync().catch(() => {});
    }
  }, [fontsLoaded, fontError]);

  if (!fontsLoaded && !fontError) return null;

  return (
    <GestureHandlerRootView style={{ flex: 1, backgroundColor: colors.bg }}>
      <SafeAreaProvider>
        <QueryClientProvider client={queryClient}>
          <AuthProvider>
            <StatusBar style="light" />
            <AuthGate />
          </AuthProvider>
        </QueryClientProvider>
      </SafeAreaProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  loadingScreen: {
    flex: 1,
    backgroundColor: colors.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
});
