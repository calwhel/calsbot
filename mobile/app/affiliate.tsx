import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TextInput,
  Pressable,
  Platform,
  Alert,
  Share,
  ActivityIndicator,
} from 'react-native';
import { Stack } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import * as Haptics from 'expo-haptics';
import * as Clipboard from 'expo-clipboard';

import { useAuth } from '@/contexts/AuthContext';
import { apiGet, apiPost, ApiError } from '@/lib/api';
import { colors, font, radius, shadow, spacing } from '@/constants/colors';
import { PrimaryButton } from '@/components/PrimaryButton';

type AffiliateStatus = 'none' | 'pending' | 'approved' | 'rejected';

type AffiliateMe = {
  is_affiliate: boolean;
  status: AffiliateStatus;
  referral_code: string | null;
  referral_url: string | null;
  sub_share_pct: number;
  fee_share_pct: number;
  application: {
    telegram?: string;
    twitter?: string;
    instagram?: string;
    youtube?: string;
    tiktok?: string;
    website?: string;
    bio?: string;
    plan?: string;
    created_at?: string;
    reviewer_note?: string;
  } | null;
  stats: { referrals: number; earnings_usd: number };
};

export default function AffiliateScreen() {
  const { uid } = useAuth();
  const qc = useQueryClient();

  const meQ = useQuery({
    queryKey: ['affiliate-me', uid],
    queryFn: () => apiGet<AffiliateMe>('/api/affiliates/me', uid!),
    enabled: !!uid,
  });

  const me = meQ.data;
  const status: AffiliateStatus = me?.status || 'none';
  const isApproved = !!me?.is_affiliate;

  // Form state
  const [telegram, setTelegram] = useState('');
  const [twitter, setTwitter] = useState('');
  const [instagram, setInstagram] = useState('');
  const [youtube, setYoutube] = useState('');
  const [tiktok, setTiktok] = useState('');
  const [website, setWebsite] = useState('');
  const [bio, setBio] = useState('');
  const [plan, setPlan] = useState('');

  // Hydrate form from any existing application so the user can edit & resubmit.
  React.useEffect(() => {
    const a = me?.application;
    if (!a) return;
    setTelegram(a.telegram || '');
    setTwitter(a.twitter || '');
    setInstagram(a.instagram || '');
    setYoutube(a.youtube || '');
    setTiktok(a.tiktok || '');
    setWebsite(a.website || '');
    setBio(a.bio || '');
    setPlan(a.plan || '');
  }, [me?.application]);

  const submitM = useMutation({
    mutationFn: () => apiPost<{ ok: true; referral_url: string }>(
      '/api/affiliates/apply',
      { telegram, twitter, instagram, youtube, tiktok, website, bio, plan },
      uid!,
    ),
    onSuccess: () => {
      if (Platform.OS !== 'web') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success).catch(() => {});
      }
      qc.invalidateQueries({ queryKey: ['affiliate-me', uid] });
      Alert.alert(
        'Application received',
        "Thanks! We review every application and reach out on Telegram within 48 hours. In the meantime your referral link is ready to share.",
      );
    },
    onError: (err: unknown) => {
      const msg = err instanceof ApiError ? err.message : 'Something went wrong. Please try again.';
      Alert.alert('Could not submit', String(msg));
    },
  });

  const onCopy = async () => {
    if (!me?.referral_url) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    await Clipboard.setStringAsync(me.referral_url);
    Alert.alert('Copied', 'Referral link copied to clipboard.');
  };

  const onShare = async () => {
    if (!me?.referral_url) return;
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    try {
      await Share.share({
        message: `Trade smarter with TradeHub — auto-trading bots, AI strategy builder, and a marketplace of pro setups. Sign up with my link: ${me.referral_url}`,
      });
    } catch {/* user cancelled */}
  };

  const onSubmit = () => {
    if (!telegram.trim()) { Alert.alert('Telegram required', 'Please share your Telegram handle so we can reach you.'); return; }
    if (bio.trim().length < 20) { Alert.alert('Bio too short', 'Tell us a bit more about yourself (20+ chars).'); return; }
    if (plan.trim().length < 30) { Alert.alert('Plan too short', 'Describe your promotion plan in a bit more detail (30+ chars).'); return; }
    if (Platform.OS !== 'web') Haptics.selectionAsync().catch(() => {});
    submitM.mutate();
  };

  return (
    <>
      <Stack.Screen options={{ title: 'Affiliate program', headerBackTitle: 'Settings' }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: colors.bg }}
        contentContainerStyle={{ paddingBottom: spacing.xxl * 2 }}
        keyboardShouldPersistTaps="handled"
      >
        {/* Hero */}
        <View style={styles.hero}>
          <View style={styles.heroBadge}>
            <Ionicons name="diamond-outline" size={14} color={colors.accent} />
            <Text style={styles.heroBadgeText}>PARTNER PROGRAM</Text>
          </View>
          <Text style={styles.heroTitle}>Earn with every trader you bring</Text>
          <Text style={styles.heroSub}>
            Join the TradeHub partner program and get a recurring share of every
            subscription and a cut of trading fees from users you refer. No caps,
            paid monthly.
          </Text>

          {/* Headline numbers */}
          <View style={styles.headlineRow}>
            <View style={styles.headlineCard}>
              <Text style={styles.headlineNum}>{me?.sub_share_pct ?? 30}%</Text>
              <Text style={styles.headlineLabel}>Subscription share</Text>
              <Text style={styles.headlineHint}>Recurring, lifetime of the user</Text>
            </View>
            <View style={styles.headlineCard}>
              <Text style={styles.headlineNum}>{me?.fee_share_pct ?? 20}%</Text>
              <Text style={styles.headlineLabel}>Trading fee share</Text>
              <Text style={styles.headlineHint}>Of every fee referred users pay</Text>
            </View>
          </View>
        </View>

        {/* Loading */}
        {meQ.isLoading ? (
          <View style={{ paddingVertical: spacing.xl }}>
            <ActivityIndicator color={colors.accent} />
          </View>
        ) : null}

        {/* Approved state — show link + share */}
        {isApproved && me?.referral_url ? (
          <View style={styles.section}>
            <View style={styles.statusBadgeApproved}>
              <Ionicons name="checkmark-circle" size={16} color={colors.positive} />
              <Text style={styles.statusBadgeApprovedText}>You're an approved partner</Text>
            </View>

            <View style={styles.linkCard}>
              <Text style={styles.linkLabel}>YOUR REFERRAL LINK</Text>
              <Text style={styles.linkUrl} numberOfLines={1}>{me.referral_url}</Text>
              <View style={styles.linkBtnRow}>
                <Pressable onPress={onCopy} style={({ pressed }) => [styles.secondaryBtn, pressed && { opacity: 0.85 }]}>
                  <Ionicons name="copy-outline" size={18} color={colors.text} />
                  <Text style={styles.secondaryBtnText}>Copy</Text>
                </Pressable>
                <Pressable onPress={onShare} style={({ pressed }) => [styles.primaryBtn, pressed && { opacity: 0.85 }]}>
                  <Ionicons name="share-outline" size={18} color="#0E0F11" />
                  <Text style={styles.primaryBtnText}>Share</Text>
                </Pressable>
              </View>
            </View>

            <View style={styles.statRow}>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>{me.stats?.referrals ?? 0}</Text>
                <Text style={styles.statLabel}>Sign-ups</Text>
              </View>
              <View style={styles.statCard}>
                <Text style={styles.statValue}>${(me.stats?.earnings_usd ?? 0).toFixed(2)}</Text>
                <Text style={styles.statLabel}>Earnings</Text>
              </View>
            </View>
          </View>
        ) : null}

        {/* Pending state */}
        {!isApproved && status === 'pending' ? (
          <View style={styles.section}>
            <View style={styles.statusBadgePending}>
              <Ionicons name="time-outline" size={16} color={colors.warning} />
              <Text style={styles.statusBadgePendingText}>Application under review</Text>
            </View>
            <Text style={styles.pendingHint}>
              We've received your application and will get back to you on Telegram
              within 48 hours. You can edit and resubmit the form below if you'd
              like to add more details.
            </Text>
            {me?.referral_url ? (
              <View style={[styles.linkCard, { marginTop: spacing.md }]}>
                <Text style={styles.linkLabel}>YOUR REFERRAL LINK (LIVE)</Text>
                <Text style={styles.linkUrl} numberOfLines={1}>{me.referral_url}</Text>
                <View style={styles.linkBtnRow}>
                  <Pressable onPress={onCopy} style={({ pressed }) => [styles.secondaryBtn, pressed && { opacity: 0.85 }]}>
                    <Ionicons name="copy-outline" size={18} color={colors.text} />
                    <Text style={styles.secondaryBtnText}>Copy</Text>
                  </Pressable>
                  <Pressable onPress={onShare} style={({ pressed }) => [styles.primaryBtn, pressed && { opacity: 0.85 }]}>
                    <Ionicons name="share-outline" size={18} color="#0E0F11" />
                    <Text style={styles.primaryBtnText}>Share</Text>
                  </Pressable>
                </View>
              </View>
            ) : null}
          </View>
        ) : null}

        {/* Rejected — let them re-apply */}
        {status === 'rejected' ? (
          <View style={styles.section}>
            <View style={styles.statusBadgeRejected}>
              <Ionicons name="close-circle" size={16} color={colors.negative} />
              <Text style={styles.statusBadgeRejectedText}>Application not approved</Text>
            </View>
            {me?.application?.reviewer_note ? (
              <Text style={styles.pendingHint}>{me.application.reviewer_note}</Text>
            ) : (
              <Text style={styles.pendingHint}>
                You can update your details below and re-submit.
              </Text>
            )}
          </View>
        ) : null}

        {/* Why join — value props */}
        {!isApproved ? (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Why partner with us</Text>
            <ValueProp
              icon="cash-outline"
              title="Recurring revenue"
              body="30% of every monthly subscription, for as long as the user stays subscribed. No expiry, no caps."
            />
            <ValueProp
              icon="trending-up-outline"
              title="Trading fee share"
              body="20% of the trading fees your referred users generate on connected exchanges."
            />
            <ValueProp
              icon="rocket-outline"
              title="Built to convert"
              body="A polished mobile app, AI-built strategies, a curated marketplace, and a real Telegram community — easy to recommend."
            />
            <ValueProp
              icon="time-outline"
              title="Paid monthly"
              body="Earnings settled in USDT (or fiat on request) on the 5th of every month, with a transparent dashboard coming soon."
            />
          </View>
        ) : null}

        {/* Application form (only for non-approved) */}
        {!isApproved ? (
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>
              {status === 'pending' ? 'Update your application' : status === 'rejected' ? 'Re-apply' : 'Apply now'}
            </Text>
            <Text style={styles.formHint}>
              Tell us a bit about yourself and how you plan to promote TradeHub.
              We approve serious creators within 48 hours.
            </Text>

            <Field label="Telegram handle" required>
              <Input value={telegram} onChangeText={setTelegram} placeholder="@yourname" autoCapitalize="none" />
            </Field>

            <Text style={styles.fieldGroupLabel}>Social links (any you have)</Text>
            <Field label="Twitter / X">
              <Input value={twitter} onChangeText={setTwitter} placeholder="https://x.com/..." autoCapitalize="none" keyboardType="url" />
            </Field>
            <Field label="Instagram">
              <Input value={instagram} onChangeText={setInstagram} placeholder="https://instagram.com/..." autoCapitalize="none" keyboardType="url" />
            </Field>
            <Field label="YouTube">
              <Input value={youtube} onChangeText={setYoutube} placeholder="https://youtube.com/@..." autoCapitalize="none" keyboardType="url" />
            </Field>
            <Field label="TikTok">
              <Input value={tiktok} onChangeText={setTiktok} placeholder="https://tiktok.com/@..." autoCapitalize="none" keyboardType="url" />
            </Field>
            <Field label="Website / blog">
              <Input value={website} onChangeText={setWebsite} placeholder="https://..." autoCapitalize="none" keyboardType="url" />
            </Field>

            <Field label="A short bio" required hint={`${bio.trim().length}/1000`}>
              <Input
                value={bio}
                onChangeText={setBio}
                placeholder="Who you are, what you trade, and the audience you've built."
                multiline
                numberOfLines={4}
                style={{ minHeight: 96, textAlignVertical: 'top' }}
              />
            </Field>

            <Field label="How will you promote TradeHub?" required hint={`${plan.trim().length}/1500`}>
              <Input
                value={plan}
                onChangeText={setPlan}
                placeholder="Channels, content cadence, audience size, and any past results."
                multiline
                numberOfLines={5}
                style={{ minHeight: 120, textAlignVertical: 'top' }}
              />
            </Field>

            <View style={{ marginTop: spacing.lg }}>
              <PrimaryButton
                label={status === 'pending' ? 'Resubmit application' : 'Submit application'}
                onPress={onSubmit}
                loading={submitM.isPending}
              />
            </View>
            <Text style={styles.disclaimer}>
              By applying, you agree to promote TradeHub honestly — no misleading
              claims, no spam, and full disclosure of the partnership where required
              by your platform.
            </Text>
          </View>
        ) : null}
      </ScrollView>
    </>
  );
}

function ValueProp({ icon, title, body }: { icon: keyof typeof Ionicons.glyphMap; title: string; body: string }) {
  return (
    <View style={styles.valueRow}>
      <View style={styles.valueIcon}>
        <Ionicons name={icon} size={20} color={colors.accent} />
      </View>
      <View style={{ flex: 1 }}>
        <Text style={styles.valueTitle}>{title}</Text>
        <Text style={styles.valueBody}>{body}</Text>
      </View>
    </View>
  );
}

function Field({
  label, required, hint, children,
}: { label: string; required?: boolean; hint?: string; children: React.ReactNode }) {
  return (
    <View style={{ marginTop: spacing.md }}>
      <View style={styles.labelRow}>
        <Text style={styles.fieldLabel}>
          {label}
          {required ? <Text style={{ color: colors.negative }}> *</Text> : null}
        </Text>
        {hint ? <Text style={styles.fieldHint}>{hint}</Text> : null}
      </View>
      {children}
    </View>
  );
}

const Input = React.forwardRef<TextInput, React.ComponentProps<typeof TextInput>>((props, ref) => (
  <TextInput
    ref={ref}
    placeholderTextColor={colors.textMute}
    {...props}
    style={[styles.input, props.style]}
  />
));
Input.displayName = 'Input';

const styles = StyleSheet.create({
  hero: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.xl,
    paddingBottom: spacing.xl,
  },
  heroBadge: {
    alignSelf: 'flex-start',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 999,
    backgroundColor: colors.accentDim,
    marginBottom: spacing.md,
  },
  heroBadgeText: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 1.4,
  },
  heroTitle: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 30,
    lineHeight: 36,
    letterSpacing: -0.6,
  },
  heroSub: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 14,
    lineHeight: 21,
    marginTop: spacing.md,
  },
  headlineRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.xl,
  },
  headlineCard: {
    flex: 1,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.lg,
    ...shadow.card,
  },
  headlineNum: {
    color: colors.accent,
    fontFamily: font.bold,
    fontSize: 32,
    letterSpacing: -0.8,
  },
  headlineLabel: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 13,
    marginTop: 4,
  },
  headlineHint: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    marginTop: 4,
  },
  section: {
    paddingHorizontal: spacing.lg,
    marginTop: spacing.xl,
  },
  sectionTitle: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 18,
    letterSpacing: -0.3,
    marginBottom: spacing.md,
  },
  // Status badges
  statusBadgeApproved: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    backgroundColor: colors.positiveDim,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    marginBottom: spacing.md,
  },
  statusBadgeApprovedText: { color: colors.positive, fontFamily: font.semibold, fontSize: 12 },
  statusBadgePending: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    backgroundColor: colors.warningDim,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    marginBottom: spacing.md,
  },
  statusBadgePendingText: { color: colors.warning, fontFamily: font.semibold, fontSize: 12 },
  statusBadgeRejected: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    alignSelf: 'flex-start',
    backgroundColor: colors.negativeDim,
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    marginBottom: spacing.md,
  },
  statusBadgeRejectedText: { color: colors.negative, fontFamily: font.semibold, fontSize: 12 },
  pendingHint: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 19,
  },
  // Link card
  linkCard: {
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.lg,
    ...shadow.card,
  },
  linkLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 1.2,
    marginBottom: 6,
  },
  linkUrl: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 15,
    marginBottom: spacing.md,
  },
  linkBtnRow: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  secondaryBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    height: 44,
    borderRadius: radius.md,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
  },
  secondaryBtnText: { color: colors.text, fontFamily: font.semibold, fontSize: 14 },
  primaryBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    height: 44,
    borderRadius: radius.md,
    backgroundColor: colors.text,
  },
  primaryBtnText: { color: '#0E0F11', fontFamily: font.semibold, fontSize: 14 },
  // Stats row
  statRow: {
    flexDirection: 'row',
    gap: spacing.md,
    marginTop: spacing.md,
  },
  statCard: {
    flex: 1,
    backgroundColor: colors.card,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.lg,
    padding: spacing.lg,
  },
  statValue: { color: colors.text, fontFamily: font.bold, fontSize: 22, letterSpacing: -0.4 },
  statLabel: { color: colors.textMute, fontFamily: font.medium, fontSize: 12, marginTop: 4 },
  // Value props
  valueRow: {
    flexDirection: 'row',
    gap: spacing.md,
    paddingVertical: spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  valueIcon: {
    width: 40, height: 40, borderRadius: radius.md,
    backgroundColor: colors.cardHi,
    borderWidth: 1, borderColor: colors.border,
    alignItems: 'center', justifyContent: 'center',
  },
  valueTitle: { color: colors.text, fontFamily: font.semibold, fontSize: 14 },
  valueBody: { color: colors.textDim, fontFamily: font.regular, fontSize: 13, lineHeight: 19, marginTop: 2 },
  // Form
  formHint: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 19,
    marginBottom: spacing.md,
  },
  fieldGroupLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 1.2,
    marginTop: spacing.lg,
  },
  labelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  fieldLabel: { color: colors.text, fontFamily: font.semibold, fontSize: 13 },
  fieldHint: { color: colors.textMute, fontFamily: font.regular, fontSize: 11 },
  input: {
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
    borderRadius: radius.md,
    paddingHorizontal: spacing.md,
    paddingVertical: 12,
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 15,
  },
  disclaimer: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 11,
    lineHeight: 16,
    marginTop: spacing.md,
    textAlign: 'center',
  },
});
