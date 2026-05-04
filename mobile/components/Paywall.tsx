import React, { useState } from 'react';
import {
  View, Text, StyleSheet, Modal, Pressable, ScrollView, ActivityIndicator, Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { colors, font, radius, spacing } from '@/constants/colors';
import { PrimaryButton } from '@/components/PrimaryButton';
import { useSubscription, isRevenueCatReady } from '@/lib/revenuecat';

type Props = {
  visible: boolean;
  onClose: () => void;
  onFallbackWeb?: () => void;
};

export function Paywall({ visible, onClose, onFallbackWeb }: Props) {
  const ready = isRevenueCatReady();

  if (!ready) {
    return (
      <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
        <Pressable style={styles.backdrop} onPress={onClose}>
          <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
            <View style={styles.handleWrap}><View style={styles.handle} /></View>
            <View style={styles.headerRow}>
              <Text style={styles.title}>Upgrade to Pro</Text>
              <Pressable onPress={onClose} hitSlop={12}><Ionicons name="close" size={22} color={colors.textDim} /></Pressable>
            </View>
            <View style={{ padding: spacing.lg, alignItems: 'center' }}>
              <View style={styles.iconCircle}>
                <Ionicons name="diamond" size={32} color={colors.text} />
              </View>
              <Text style={styles.proTitle}>TradeHub Pro</Text>
              <Text style={styles.proDesc}>
                In-app subscriptions are being set up. For now, you can upgrade on the web.
              </Text>
              {onFallbackWeb ? (
                <View style={{ marginTop: spacing.lg, width: '100%' }}>
                  <PrimaryButton label="Upgrade on web" onPress={onFallbackWeb} />
                </View>
              ) : null}
            </View>
          </Pressable>
        </Pressable>
      </Modal>
    );
  }

  return <PaywallWithRC visible={visible} onClose={onClose} />;
}

function PaywallWithRC({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const { offerings, purchase, isPurchasing, restore, isRestoring } = useSubscription();
  const [error, setError] = useState<string | null>(null);

  const currentOffering = offerings?.current;
  const pkg = currentOffering?.availablePackages?.[0];
  const price = pkg?.product?.priceString || '$9.99/mo';

  const onPurchase = async () => {
    if (!pkg) return;
    setError(null);
    try {
      await purchase(pkg);
      onClose();
    } catch (e: any) {
      if (e?.userCancelled) return;
      setError(e?.message || 'Purchase failed. Please try again.');
    }
  };

  const onRestore = async () => {
    setError(null);
    try {
      await restore();
      onClose();
    } catch (e: any) {
      setError(e?.message || 'Could not restore purchases.');
    }
  };

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose} statusBarTranslucent>
      <Pressable style={styles.backdrop} onPress={onClose}>
        <Pressable style={styles.sheet} onPress={(e) => e.stopPropagation()}>
          <View style={styles.handleWrap}><View style={styles.handle} /></View>
          <View style={styles.headerRow}>
            <Text style={styles.title}>Upgrade to Pro</Text>
            <Pressable onPress={onClose} hitSlop={12}><Ionicons name="close" size={22} color={colors.textDim} /></Pressable>
          </View>

          <ScrollView style={{ maxHeight: 500 }} contentContainerStyle={{ padding: spacing.lg }}>
            <View style={{ alignItems: 'center', marginBottom: spacing.lg }}>
              <View style={styles.iconCircle}>
                <Ionicons name="diamond" size={32} color={colors.text} />
              </View>
              <Text style={styles.proTitle}>TradeHub Pro</Text>
              <Text style={styles.proPrice}>{price}</Text>
            </View>

            {[
              'Unlimited backtests with full analytics',
              'Live trade execution on connected exchanges',
              'AI-powered strategy suggestions',
              'Priority signal scanning',
              'Marketplace premium strategies',
            ].map((b) => (
              <View key={b} style={styles.bulletRow}>
                <Ionicons name="checkmark-circle" size={18} color="#3FB68B" />
                <Text style={styles.bulletText}>{b}</Text>
              </View>
            ))}

            {error ? (
              <View style={styles.errorBox}>
                <Text style={styles.errorText}>{error}</Text>
              </View>
            ) : null}

            <View style={{ marginTop: spacing.xl }}>
              <PrimaryButton
                label={isPurchasing ? 'Processing...' : `Subscribe for ${price}`}
                onPress={onPurchase}
                loading={isPurchasing}
                disabled={!pkg || isPurchasing}
              />
            </View>

            <Pressable onPress={onRestore} disabled={isRestoring} style={styles.restoreBtn}>
              {isRestoring
                ? <ActivityIndicator size="small" color={colors.textDim} />
                : <Text style={styles.restoreText}>Restore purchases</Text>}
            </Pressable>

            <Text style={styles.legalText}>
              Payment will be charged to your {Platform.OS === 'ios' ? 'Apple ID' : 'Google Play'} account.
              Subscription automatically renews unless cancelled at least 24 hours before the end of the current period.
              Manage subscriptions in your device settings.
            </Text>
          </ScrollView>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, backgroundColor: 'rgba(0,0,0,0.55)', justifyContent: 'flex-end' },
  sheet: {
    backgroundColor: colors.bg,
    borderTopLeftRadius: 24, borderTopRightRadius: 24,
    paddingHorizontal: spacing.md, paddingTop: 6, paddingBottom: spacing.lg,
    borderTopWidth: 1, borderColor: colors.border,
  },
  handleWrap: { alignItems: 'center', paddingVertical: 8 },
  handle: { width: 40, height: 4, borderRadius: 2, backgroundColor: '#444' },
  headerRow: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between',
    paddingBottom: spacing.sm, paddingHorizontal: spacing.sm,
  },
  title: { color: colors.text, fontFamily: font.bold, fontSize: 18 },
  iconCircle: {
    width: 64, height: 64, borderRadius: 32,
    backgroundColor: 'rgba(63,182,139,0.12)',
    borderWidth: 1, borderColor: 'rgba(63,182,139,0.24)',
    alignItems: 'center', justifyContent: 'center',
    marginBottom: spacing.md,
  },
  proTitle: { color: colors.text, fontFamily: font.black, fontSize: 22, letterSpacing: -0.4 },
  proPrice: { color: colors.textDim, fontFamily: font.semibold, fontSize: 16, marginTop: 4 },
  proDesc: {
    color: colors.textMute, fontFamily: font.regular, fontSize: 13,
    textAlign: 'center', marginTop: spacing.sm, lineHeight: 18,
  },
  bulletRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 12 },
  bulletText: { color: colors.text, fontFamily: font.medium, fontSize: 14, flex: 1 },
  errorBox: {
    marginTop: spacing.md, padding: 12, borderRadius: radius.md,
    backgroundColor: 'rgba(229,72,77,0.08)', borderWidth: 1, borderColor: 'rgba(229,72,77,0.24)',
  },
  errorText: { color: '#E5484D', fontFamily: font.medium, fontSize: 13 },
  restoreBtn: { alignItems: 'center', paddingVertical: 14 },
  restoreText: { color: colors.textDim, fontFamily: font.semibold, fontSize: 13 },
  legalText: {
    color: colors.textMute, fontFamily: font.regular, fontSize: 10,
    textAlign: 'center', lineHeight: 14,
  },
});
