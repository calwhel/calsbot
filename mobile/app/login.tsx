import React, { useCallback, useState } from 'react';
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, Pressable, Linking,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '@/contexts/AuthContext';
import { ApiError } from '@/lib/api';
import { PrimaryButton } from '@/components/PrimaryButton';
import { colors, radius, spacing } from '@/constants/colors';

export default function LoginScreen() {
  const insets = useSafeAreaInsets();
  const { signIn } = useAuth();
  const [uid, setUid] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = useCallback(async () => {
    setError(null);
    if (!uid.trim()) {
      setError('Enter your UID to continue.');
      return;
    }
    setLoading(true);
    try {
      await signIn(uid);
      // AuthGate redirects on user state change.
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.status === 403 ? 'That UID was not recognised.' : `Login failed (${e.status}).`);
      } else {
        setError('Could not reach the server. Check your connection and try again.');
      }
    } finally {
      setLoading(false);
    }
  }, [uid, signIn]);

  const openSite = () => {
    Linking.openURL('https://tradehub.markets').catch(() => {});
  };

  return (
    <KeyboardAvoidingView
      style={styles.root}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <ScrollView
        contentContainerStyle={[styles.scroll, { paddingTop: insets.top + 40, paddingBottom: insets.bottom + 24 }]}
        keyboardShouldPersistTaps="handled"
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.brand}>
          <View style={styles.logoBox}>
            <Ionicons name="trending-up" size={28} color={colors.accent} />
          </View>
          <Text style={styles.brandTitle}>TradeHub</Text>
          <Text style={styles.brandSub}>Your strategies. Your edge. In your pocket.</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.label}>Your UID</Text>
          <TextInput
            value={uid}
            onChangeText={setUid}
            placeholder="TH-XXXXXXXX"
            placeholderTextColor={colors.textMute}
            autoCapitalize="characters"
            autoCorrect={false}
            autoComplete="off"
            style={styles.input}
            editable={!loading}
            returnKeyType="go"
            onSubmitEditing={onSubmit}
            maxLength={20}
          />

          {error ? (
            <View style={styles.errorBox}>
              <Ionicons name="alert-circle" size={16} color={colors.negative} />
              <Text style={styles.errorText}>{error}</Text>
            </View>
          ) : null}

          <View style={{ height: spacing.lg }} />
          <PrimaryButton label="Sign in" onPress={onSubmit} loading={loading} />

          <Pressable onPress={openSite} style={styles.helpLink}>
            <Ionicons name="help-circle-outline" size={14} color={colors.textDim} />
            <Text style={styles.helpText}>
              Don't have a UID? Sign up at tradehub.markets
            </Text>
          </Pressable>
        </View>

        <Text style={styles.footer}>
          Your UID is stored securely on this device only.
        </Text>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: colors.bg },
  scroll: {
    flexGrow: 1,
    paddingHorizontal: spacing.xl,
    justifyContent: 'center',
  },
  brand: {
    alignItems: 'center',
    marginBottom: spacing.xxl,
  },
  logoBox: {
    width: 64, height: 64, borderRadius: radius.lg,
    backgroundColor: colors.accentDim,
    alignItems: 'center', justifyContent: 'center',
    marginBottom: spacing.md,
  },
  brandTitle: {
    color: colors.text,
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  brandSub: {
    color: colors.textDim,
    fontSize: 14,
    marginTop: 4,
    textAlign: 'center',
  },
  card: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.xl,
  },
  label: {
    color: colors.textDim,
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
  input: {
    backgroundColor: colors.bgElev,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    color: colors.text,
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 1.2,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.md,
    backgroundColor: colors.negativeDim,
    padding: 10,
    borderRadius: radius.md,
  },
  errorText: {
    color: colors.negative,
    fontSize: 13,
    flex: 1,
  },
  helpLink: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.lg,
    justifyContent: 'center',
  },
  helpText: {
    color: colors.textDim,
    fontSize: 12,
  },
  footer: {
    color: colors.textMute,
    fontSize: 11,
    textAlign: 'center',
    marginTop: spacing.xl,
  },
});
