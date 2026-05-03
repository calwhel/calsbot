import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import { InfoCard } from '@/components/InfoCard';
import { colors, font, radius, spacing } from '@/constants/colors';

type Step = {
  icon: React.ComponentProps<typeof Ionicons>['name'];
  iconColor: string;
  title: string;
  body: string;
};

const STEPS: Step[] = [
  {
    icon: 'scan-outline',
    iconColor: colors.textDim,
    title: 'We scan the markets',
    body: 'Every few minutes our engine checks your trigger conditions across the coins you watch.',
  },
  {
    icon: 'flash-outline',
    iconColor: colors.textDim,
    title: 'The strategy fires',
    body: "When all conditions match, a paper trade opens at the live price — no exchange account needed to test.",
  },
  {
    icon: 'notifications-outline',
    iconColor: colors.textDim,
    title: 'You get notified',
    body: 'A push notification + Telegram message lands instantly so you can copy the trade or just watch.',
  },
  {
    icon: 'flag-outline',
    iconColor: colors.positive,
    title: 'We track the result',
    body: 'Take-profit and stop-loss are watched 24/7. The closed trade lands in your history with full P&L.',
  },
];

/**
 * Educational explainer rendered on the strategy detail screen so a brand-new
 * user understands what "active" actually means and where notifications come
 * from. Static copy — no API.
 */
export function AutomationCard() {
  return (
    <InfoCard label="What happens when this is active" icon="play-circle-outline" iconColor={colors.violet}>
      <View style={{ paddingTop: 4 }}>
        {STEPS.map((step, i) => (
          <View key={`step-${i}`} style={styles.row}>
            <View style={styles.left}>
              <View style={styles.bubble}>
                <Ionicons name={step.icon} size={16} color={step.iconColor} />
              </View>
              {i < STEPS.length - 1 ? <View style={styles.connector} /> : null}
            </View>
            <View style={styles.right}>
              <Text style={styles.stepTitle}>
                <Text style={[styles.stepNum, { color: step.iconColor }]}>{i + 1}.</Text> {step.title}
              </Text>
              <Text style={styles.stepBody}>{step.body}</Text>
            </View>
          </View>
        ))}
      </View>
    </InfoCard>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    paddingTop: 10,
  },
  left: {
    width: 36,
    alignItems: 'center',
  },
  bubble: {
    width: 30,
    height: 30,
    borderRadius: radius.md,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.cardHi,
    borderWidth: 1,
    borderColor: colors.border,
  },
  connector: {
    flex: 1,
    width: 1.5,
    backgroundColor: colors.divider,
    marginTop: 4,
    marginBottom: -4,
  },
  right: {
    flex: 1,
    paddingLeft: 12,
    paddingBottom: spacing.md,
  },
  stepTitle: {
    color: colors.text,
    fontFamily: font.semibold,
    fontSize: 14,
    marginTop: 4,
  },
  stepNum: { fontFamily: font.black },
  stepBody: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 12.5,
    marginTop: 2,
    lineHeight: 18,
  },
});
