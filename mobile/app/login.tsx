import React, { useCallback, useState } from 'react';
import {
  View, Text, TextInput, StyleSheet, KeyboardAvoidingView,
  Platform, ScrollView, Pressable, Linking,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as AppleAuthentication from 'expo-apple-authentication';
import { useAuth } from '@/contexts/AuthContext';
import { ApiError } from '@/lib/api';
import { PrimaryButton } from '@/components/PrimaryButton';
import { Logo } from '@/components/Logo';
import { AmbientBg } from '@/components/AmbientBg';
import { GradientCard } from '@/components/GradientCard';
import { RiskDisclaimer } from '@/components/RiskDisclaimer';
import { colors, font, radius, spacing } from '@/constants/colors';

type Mode = 'uid' | 'email';

export default function LoginScreen() {
  const insets = useSafeAreaInsets();
  const { signIn, signInEmail, signInApple } = useAuth();
  const [mode, setMode] = useState<Mode>('uid');
  const [uid, setUid] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onSubmit = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      if (mode === 'uid') {
        if (!uid.trim()) { setError('Enter your UID to continue.'); return; }
        await signIn(uid);
      } else {
        if (!email.trim() || !password) {
          setError('Enter your email and password.');
          return;
        }
        await signInEmail(email, password);
      }
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.message || (e.status === 403 ? 'Login failed.' : `Login failed (${e.status}).`));
      } else {
        setError('Could not reach the server. Check your connection and try again.');
      }
    } finally {
      setLoading(false);
    }
  }, [mode, uid, email, password, signIn, signInEmail]);

  const switchMode = (m: Mode) => {
    setMode(m);
    setError(null);
  };

  const openSite = () => {
    Linking.openURL('https://tradehub.markets').catch(() => {});
  };

  return (
    <View style={styles.root}>
      <AmbientBg variant="duo" />
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      >
        <ScrollView
          contentContainerStyle={[styles.scroll, { paddingTop: insets.top + 32, paddingBottom: insets.bottom + 24 }]}
          keyboardShouldPersistTaps="handled"
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.brand}>
            <Logo size={76} />
            <Text style={styles.brandTitle}>TradeHub</Text>
            <Text style={styles.brandSub}>Your strategies. Your edge. In your pocket.</Text>
          </View>

          <GradientCard style={styles.card}>
            <View style={styles.cardInner}>
              {/* Mode toggle */}
              <View style={styles.tabs}>
                <ModeTab label="UID" active={mode === 'uid'} onPress={() => switchMode('uid')} />
                <ModeTab label="Email" active={mode === 'email'} onPress={() => switchMode('email')} />
              </View>

              {mode === 'uid' ? (
                <>
                  <Text style={styles.label}>Your UID</Text>
                  <TextInput
                    value={uid}
                    onChangeText={setUid}
                    placeholder="TH-XXXXXXXX"
                    placeholderTextColor={colors.textMute}
                    autoCapitalize="characters"
                    autoCorrect={false}
                    autoComplete="off"
                    style={[styles.input, styles.inputUid]}
                    editable={!loading}
                    returnKeyType="go"
                    onSubmitEditing={onSubmit}
                    maxLength={20}
                  />
                </>
              ) : (
                <>
                  <Text style={styles.label}>Email</Text>
                  <TextInput
                    value={email}
                    onChangeText={setEmail}
                    placeholder="you@example.com"
                    placeholderTextColor={colors.textMute}
                    autoCapitalize="none"
                    autoCorrect={false}
                    autoComplete="email"
                    keyboardType="email-address"
                    style={styles.input}
                    editable={!loading}
                    returnKeyType="next"
                  />
                  <View style={{ height: spacing.md }} />
                  <Text style={styles.label}>Password</Text>
                  <TextInput
                    value={password}
                    onChangeText={setPassword}
                    placeholder="••••••••"
                    placeholderTextColor={colors.textMute}
                    autoCapitalize="none"
                    autoCorrect={false}
                    autoComplete="current-password"
                    secureTextEntry
                    style={styles.input}
                    editable={!loading}
                    returnKeyType="go"
                    onSubmitEditing={onSubmit}
                  />
                </>
              )}

              {error ? (
                <View style={styles.errorBox}>
                  <Ionicons name="alert-circle" size={16} color={colors.negative} />
                  <Text style={styles.errorText}>{error}</Text>
                </View>
              ) : null}

              <View style={{ height: spacing.lg }} />
              <PrimaryButton label="Sign in" onPress={onSubmit} loading={loading} />

              {Platform.OS === 'ios' ? (
                <View style={{ marginTop: spacing.lg }}>
                  <View style={styles.dividerRow}>
                    <View style={styles.dividerLine} />
                    <Text style={styles.dividerText}>or</Text>
                    <View style={styles.dividerLine} />
                  </View>
                  <AppleAuthentication.AppleAuthenticationButton
                    buttonType={AppleAuthentication.AppleAuthenticationButtonType.SIGN_IN}
                    buttonStyle={AppleAuthentication.AppleAuthenticationButtonStyle.WHITE}
                    cornerRadius={radius.md}
                    style={styles.appleBtn}
                    onPress={async () => {
                      try {
                        setError(null);
                        setLoading(true);
                        const credential = await AppleAuthentication.signInAsync({
                          requestedScopes: [
                            AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
                            AppleAuthentication.AppleAuthenticationScope.EMAIL,
                          ],
                        });
                        if (!credential.identityToken) {
                          setError('Apple sign-in failed — no identity token received.');
                          return;
                        }
                        const fullName = [credential.fullName?.givenName, credential.fullName?.familyName]
                          .filter(Boolean).join(' ') || null;
                        await signInApple(credential.identityToken, fullName, credential.email);
                      } catch (e: any) {
                        if (e?.code === 'ERR_REQUEST_CANCELED') return;
                        if (e instanceof ApiError) {
                          setError(e.message || 'Apple sign-in failed.');
                        } else {
                          setError('Apple sign-in failed. Please try again.');
                        }
                      } finally {
                        setLoading(false);
                      }
                    }}
                  />
                </View>
              ) : null}

              <Pressable onPress={openSite} style={styles.helpLink}>
                <Ionicons name="help-circle-outline" size={14} color={colors.textDim} />
                <Text style={styles.helpText}>
                  {mode === 'uid'
                    ? "Don't have a UID? Sign up at tradehub.markets"
                    : "No account? Sign up at tradehub.markets"}
                </Text>
              </Pressable>
            </View>
          </GradientCard>

          <RiskDisclaimer />

          <Text style={styles.footer}>
            Your credentials are stored securely on this device.
          </Text>

          <View style={styles.legalRow}>
            <Pressable onPress={() => Linking.openURL('https://tradehubmarkets.com/privacy').catch(() => {})}>
              <Text style={styles.legalLink}>Privacy Policy</Text>
            </Pressable>
            <Text style={styles.legalSep}>|</Text>
            <Pressable onPress={() => Linking.openURL('https://tradehubmarkets.com/terms').catch(() => {})}>
              <Text style={styles.legalLink}>Terms</Text>
            </Pressable>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

function ModeTab({ label, active, onPress }: { label: string; active: boolean; onPress: () => void }) {
  return (
    <Pressable
      onPress={onPress}
      style={({ pressed }) => [
        styles.tab,
        active && styles.tabActive,
        pressed && !active && { opacity: 0.7 },
      ]}
    >
      <Text style={[styles.tabText, active && styles.tabTextActive]}>{label}</Text>
    </Pressable>
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
  brandTitle: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 34,
    letterSpacing: -0.8,
    marginTop: spacing.lg,
  },
  brandSub: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    marginTop: 4,
    textAlign: 'center',
  },
  card: {},
  cardInner: {
    padding: spacing.xl,
  },
  tabs: {
    flexDirection: 'row',
    backgroundColor: colors.bgElev,
    borderRadius: radius.pill,
    padding: 4,
    marginBottom: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
  },
  tab: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: radius.pill,
    alignItems: 'center',
  },
  tabActive: {
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.borderHi,
  },
  tabText: {
    color: colors.textDim,
    fontFamily: font.semibold,
    fontSize: 13,
  },
  tabTextActive: {
    color: colors.text,
    fontFamily: font.bold,
  },
  label: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 11,
    letterSpacing: 0.7,
    textTransform: 'uppercase',
    marginBottom: spacing.sm,
  },
  input: {
    backgroundColor: colors.bg,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    color: colors.text,
    fontFamily: font.medium,
    fontSize: 16,
    paddingHorizontal: 14,
    paddingVertical: 14,
  },
  inputUid: {
    fontFamily: font.bold,
    letterSpacing: 1.2,
  },
  errorBox: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.md,
    backgroundColor: colors.negativeDim,
    padding: 10,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: 'rgba(248,113,113,0.32)',
  },
  errorText: {
    color: colors.negative,
    fontFamily: font.medium,
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
    fontFamily: font.regular,
    fontSize: 12,
  },
  footer: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    textAlign: 'center',
    marginTop: spacing.xl,
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border,
  },
  dividerText: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 12,
    marginHorizontal: 12,
  },
  appleBtn: {
    width: '100%',
    height: 48,
  },
  legalRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 8,
    marginTop: spacing.md,
  },
  legalLink: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 11,
    textDecorationLine: 'underline',
  },
  legalSep: {
    color: colors.textMute,
    fontSize: 11,
  },
});
