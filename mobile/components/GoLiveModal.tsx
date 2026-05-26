import React, { useState } from 'react';
import {
  Modal, View, Text, StyleSheet, Pressable, ScrollView,
  Linking, SafeAreaView,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';
import * as Haptics from 'expo-haptics';
import { colors, font, radius, spacing } from '@/constants/colors';

export type GoLiveBroker = 'bitunix' | 'ctrader';

interface Props {
  visible: boolean;
  onClose: () => void;
  defaultBroker?: GoLiveBroker;
}

const BITUNIX_REFERRAL = 'https://www.bitunix.com/register?vipCode=tradehubsave';
const FP_REFERRAL = 'https://portal.fpmarkets.com/register?fpm-affiliate-utm-source=IB&fpm-affiliate-agt=66940&_gl=1*ch7bdc*_gcl_au*MjA1NjIwMDU2NS4xNzc3MzA4MTUxLjk3OTE1MDYzMC4xNzk4MTU0ODkuMTc3OTgxNTYyNA..*_ga*MTI5NzM3NjM0MC4xNzU2Mjg2NDc0*_ga_GRFVC7S1MC*czE3Nzk4MTU0ODUkbzUkZzEkdDE3Nzk4MTY3MjUkajQ3JGwwJGg0MTI0MzI3MzI.';

type Step = { icon: string; title: string; body: string; cta?: { label: string; action: () => void } };

export function GoLiveModal({ visible, onClose, defaultBroker = 'bitunix' }: Props) {
  const router = useRouter();
  const [tab, setTab] = useState<GoLiveBroker>(defaultBroker);

  const open = (url: string) => Linking.openURL(url).catch(() => {});

  const navigateTo = (path: string) => {
    onClose();
    setTimeout(() => router.push(path as any), 300);
  };

  const bitunixSteps: Step[] = [
    {
      icon: 'person-add-outline',
      title: 'Sign up via TradeHub',
      body: 'Create a Bitunix account through our affiliate link. This is required — accounts not registered under TradeHub cannot access live trading.',
      cta: { label: 'Open sign-up page', action: () => open(BITUNIX_REFERRAL) },
    },
    {
      icon: 'copy-outline',
      title: 'Copy your Bitunix UID',
      body: 'In Bitunix, go to Profile → UID. It\'s a long numeric ID. Copy it to your clipboard.',
    },
    {
      icon: 'key-outline',
      title: 'Create an API key',
      body: 'In Bitunix, go to Account → API Management → Create API Key. Enable Trade permission only. Copy the key and secret.',
    },
    {
      icon: 'settings-outline',
      title: 'Connect in Settings',
      body: 'Paste your UID and API key/secret in Settings → Brokers → Bitunix. The affiliate check runs automatically — live trading unlocks within a few minutes.',
      cta: { label: 'Go to Bitunix settings', action: () => navigateTo('/bitunix') },
    },
  ];

  const ctraderSteps: Step[] = [
    {
      icon: 'person-add-outline',
      title: 'Open an FP Markets account',
      body: 'Sign up via the TradeHub introducing broker link. Your account must be linked to us to qualify for live trading on TradeHub.',
      cta: { label: 'Open FP Markets account', action: () => open(FP_REFERRAL) },
    },
    {
      icon: 'desktop-outline',
      title: 'Choose cTrader platform',
      body: 'During FP Markets sign-up, select cTrader as your trading platform. This is what gives TradeHub API access to place orders.',
    },
    {
      icon: 'shield-checkmark-outline',
      title: 'Connect via OAuth',
      body: 'In Settings → Brokers → cTrader, tap Connect. You\'ll be taken to Spotware\'s login page — sign in with your FP Markets cTrader ID and authorise TradeHub.',
      cta: { label: 'Go to cTrader settings', action: () => navigateTo('/ctrader') },
    },
    {
      icon: 'flash-outline',
      title: 'Live forex & indices unlocked',
      body: 'Once connected, any strategy set to Live mode with a forex or index asset class will route real orders through your FP Markets account instantly.',
    },
  ];

  const steps = tab === 'bitunix' ? bitunixSteps : ctraderSteps;

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}
    >
      <Pressable style={styles.backdrop} onPress={onClose} />
      <SafeAreaView style={styles.sheet} pointerEvents="box-none">
        <View style={styles.handle} />

        {/* Header */}
        <View style={styles.header}>
          <View>
            <Text style={styles.title}>Go Live — Setup Guide</Text>
            <Text style={styles.subtitle}>Follow these steps to unlock real trading</Text>
          </View>
          <Pressable onPress={onClose} hitSlop={12} style={styles.closeBtn}>
            <Ionicons name="close" size={20} color={colors.textDim} />
          </Pressable>
        </View>

        {/* Broker tabs */}
        <View style={styles.tabRow}>
          <Pressable
            style={[styles.tab, tab === 'bitunix' && styles.tabActive]}
            onPress={async () => { await Haptics.selectionAsync(); setTab('bitunix'); }}
          >
            <Ionicons
              name="trending-up-outline"
              size={14}
              color={tab === 'bitunix' ? '#000' : colors.textDim}
            />
            <Text style={[styles.tabText, tab === 'bitunix' && styles.tabTextActive]}>
              Bitunix · Crypto
            </Text>
          </Pressable>
          <Pressable
            style={[styles.tab, tab === 'ctrader' && styles.tabActive]}
            onPress={async () => { await Haptics.selectionAsync(); setTab('ctrader'); }}
          >
            <Ionicons
              name="globe-outline"
              size={14}
              color={tab === 'ctrader' ? '#000' : colors.textDim}
            />
            <Text style={[styles.tabText, tab === 'ctrader' && styles.tabTextActive]}>
              cTrader · Forex &amp; Indices
            </Text>
          </Pressable>
        </View>

        {/* Steps */}
        <ScrollView
          style={styles.scroll}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Affiliate notice */}
          <View style={styles.noticeCard}>
            <Ionicons name="information-circle-outline" size={15} color={colors.warning} />
            <Text style={styles.noticeText}>
              {tab === 'bitunix'
                ? 'You must register on Bitunix through the TradeHub affiliate link (step 1) — existing accounts signed up elsewhere cannot go live.'
                : 'Your FP Markets account must be opened via the TradeHub introducing broker link — existing accounts opened independently cannot go live.'}
            </Text>
          </View>

          {steps.map((step, i) => (
            <View key={i} style={styles.stepRow}>
              {/* Number + line */}
              <View style={styles.stepLeft}>
                <View style={styles.stepNum}>
                  <Text style={styles.stepNumText}>{i + 1}</Text>
                </View>
                {i < steps.length - 1 && <View style={styles.stepLine} />}
              </View>

              {/* Content */}
              <View style={styles.stepContent}>
                <View style={styles.stepTitleRow}>
                  <Ionicons name={step.icon as any} size={15} color={colors.textDim} />
                  <Text style={styles.stepTitle}>{step.title}</Text>
                </View>
                <Text style={styles.stepBody}>{step.body}</Text>
                {step.cta && (
                  <Pressable
                    style={styles.ctaBtn}
                    onPress={async () => { await Haptics.selectionAsync(); step.cta!.action(); }}
                  >
                    <Ionicons name="open-outline" size={13} color={colors.positive} />
                    <Text style={styles.ctaBtnText}>{step.cta.label}</Text>
                  </Pressable>
                )}
              </View>
            </View>
          ))}

          {/* Bottom CTA */}
          <Pressable
            style={styles.mainCta}
            onPress={async () => {
              await Haptics.selectionAsync();
              navigateTo(tab === 'bitunix' ? '/bitunix' : '/ctrader');
            }}
          >
            <Text style={styles.mainCtaText}>
              Open {tab === 'bitunix' ? 'Bitunix' : 'cTrader'} setup →
            </Text>
          </Pressable>

          <Text style={styles.paperNote}>
            Paper trading works without any connection — strategies run as paper until you're verified.
          </Text>
        </ScrollView>
      </SafeAreaView>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  sheet: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    backgroundColor: colors.card,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    borderWidth: 1,
    borderColor: colors.borderHi,
    maxHeight: '88%',
  },
  handle: {
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
    alignSelf: 'center',
    marginTop: 10,
    marginBottom: 4,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  title: {
    fontSize: 17,
    fontFamily: font.bold,
    color: colors.text,
  },
  subtitle: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    marginTop: 2,
  },
  closeBtn: {
    padding: 4,
  },
  tabRow: {
    flexDirection: 'row',
    marginHorizontal: spacing.lg,
    marginBottom: spacing.md,
    backgroundColor: colors.cardHi,
    borderRadius: radius.md,
    padding: 3,
    gap: 3,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 5,
    paddingVertical: 8,
    borderRadius: radius.sm,
  },
  tabActive: {
    backgroundColor: colors.positive,
  },
  tabText: {
    fontSize: 12,
    fontFamily: font.semibold,
    color: colors.textDim,
  },
  tabTextActive: {
    color: '#000',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xl + 16,
  },
  noticeCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    backgroundColor: `${colors.warning}12`,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: `${colors.warning}30`,
    padding: spacing.md,
    marginBottom: spacing.lg,
  },
  noticeText: {
    flex: 1,
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 17,
  },
  stepRow: {
    flexDirection: 'row',
    gap: 14,
    marginBottom: 0,
  },
  stepLeft: {
    alignItems: 'center',
    width: 28,
  },
  stepNum: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.borderHi,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepNumText: {
    fontSize: 12,
    fontFamily: font.bold,
    color: colors.text,
  },
  stepLine: {
    width: 1,
    flex: 1,
    backgroundColor: colors.border,
    marginTop: 4,
    marginBottom: 4,
    minHeight: 20,
  },
  stepContent: {
    flex: 1,
    paddingBottom: spacing.lg,
  },
  stepTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 4,
  },
  stepTitle: {
    fontSize: 13,
    fontFamily: font.semibold,
    color: colors.text,
  },
  stepBody: {
    fontSize: 12,
    fontFamily: font.regular,
    color: colors.textDim,
    lineHeight: 18,
  },
  ctaBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 8,
    paddingVertical: 6,
    paddingHorizontal: 10,
    borderRadius: radius.sm,
    borderWidth: 1,
    borderColor: `${colors.positive}50`,
    alignSelf: 'flex-start',
    backgroundColor: `${colors.positive}10`,
  },
  ctaBtnText: {
    fontSize: 12,
    fontFamily: font.semibold,
    color: colors.positive,
  },
  mainCta: {
    backgroundColor: colors.positive,
    borderRadius: radius.md,
    paddingVertical: 13,
    alignItems: 'center',
    marginBottom: spacing.md,
  },
  mainCtaText: {
    fontSize: 14,
    fontFamily: font.bold,
    color: '#000',
  },
  paperNote: {
    fontSize: 11,
    fontFamily: font.regular,
    color: colors.textMute,
    textAlign: 'center',
    lineHeight: 16,
  },
});
