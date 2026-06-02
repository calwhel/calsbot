import React, { useMemo } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, Pressable } from 'react-native';
import { useQuery } from '@tanstack/react-query';
import { Ionicons } from '@expo/vector-icons';
import { Stack, useRouter } from 'expo-router';

import { Screen } from '@/components/Screen';
import { SectionLabel } from '@/components/SectionLabel';
import { EmptyState } from '@/components/EmptyState';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { getPortfolioReview, type PortfolioReview } from '@/lib/api';

type Section = { heading: string | null; body: string };

const HEADINGS = ['WORKING', 'BLEEDING', 'OVERLAP', 'GAPS', 'NEXT MOVE'];

const HEADING_META: Record<string, { tone: 'positive' | 'warning' | 'accent' | 'violet'; icon: keyof typeof Ionicons.glyphMap }> = {
  WORKING:    { tone: 'positive', icon: 'trending-up' },
  BLEEDING:   { tone: 'warning',  icon: 'trending-down' },
  OVERLAP:    { tone: 'warning',  icon: 'copy-outline' },
  GAPS:       { tone: 'accent',   icon: 'add-circle-outline' },
  'NEXT MOVE':{ tone: 'violet',   icon: 'flag-outline' },
};

/** Split the AI prose into labelled sections keyed off the known headings. */
function parseReview(text: string): Section[] {
  const lines = (text || '').split('\n');
  const out: Section[] = [];
  let cur: Section = { heading: null, body: '' };
  const headingRe = new RegExp(`^\\s*\\**\\s*(${HEADINGS.join('|')})\\b\\s*[:\\-—]?\\s*`, 'i');

  for (const raw of lines) {
    const m = raw.match(headingRe);
    if (m) {
      if (cur.heading !== null || cur.body.trim()) out.push(cur);
      const rest = raw.replace(headingRe, '').trim();
      cur = { heading: m[1].toUpperCase(), body: rest ? rest + '\n' : '' };
    } else {
      cur.body += raw + '\n';
    }
  }
  if (cur.heading !== null || cur.body.trim()) out.push(cur);
  return out.filter((s) => s.heading !== null || s.body.trim());
}

function cleanBody(s: string): string {
  return s.replace(/\*\*/g, '').trim();
}

export default function PortfolioScreen() {
  const { uid } = useAuth();
  const router = useRouter();

  const q = useQuery({
    queryKey: ['portfolio-review', uid],
    queryFn: () => getPortfolioReview(uid as string),
    enabled: !!uid,
    staleTime: 60_000,
  });

  const data: PortfolioReview | undefined = q.data;
  const sections = useMemo(
    () => (data?.review ? parseReview(data.review) : []),
    [data?.review],
  );

  return (
    <Screen
      title="Portfolio Review"
      subtitle="AI review of your entire strategy book."
      refreshing={q.isFetching && !q.isLoading}
      onRefresh={() => q.refetch()}
      ambient="duo"
    >
      <Stack.Screen options={{ title: 'Portfolio' }} />

      {q.isLoading ? (
        <View style={styles.loading}>
          <ActivityIndicator color={colors.accent} size="large" />
          <Text style={styles.loadingText}>Analyzing your entire portfolio…</Text>
        </View>
      ) : q.isError ? (
        <EmptyState
          icon="cloud-offline-outline"
          title="Couldn't load your portfolio review"
          hint="Pull down to retry."
        />
      ) : data?.pro_required ? (
        <View style={styles.proCard}>
          <Ionicons name="lock-closed" size={30} color={colors.accent} />
          <Text style={styles.proTitle}>Portfolio Review is a Pro feature</Text>
          <Text style={styles.proSub}>
            Get an AI review of your whole strategy book — what's working, what's bleeding,
            overlaps, and gaps to fill.
          </Text>
        </View>
      ) : data && data.n_strategies === 0 ? (
        <EmptyState
          icon="albums-outline"
          title="No strategies yet"
          hint="Build your first strategy, paper-test it, then come back for a full review."
          ctaLabel="Build a strategy"
          onCta={() => router.push('/build' as any)}
        />
      ) : data?.review ? (
        <>
          <View style={styles.overviewCard}>
            <Ionicons name="bar-chart" size={16} color={colors.accent} />
            <Text style={styles.overviewText}>
              Reviewing {data.n_strategies} strateg{data.n_strategies === 1 ? 'y' : 'ies'} across your book.
            </Text>
          </View>

          {sections.map((sec, i) => {
            const meta = sec.heading ? HEADING_META[sec.heading] : undefined;
            return (
              <View key={`sec-${i}`} style={styles.section}>
                {sec.heading ? (
                  <SectionLabel label={sec.heading} tone={meta?.tone ?? 'accent'} />
                ) : null}
                <View style={styles.bodyCard}>
                  {sec.heading && meta ? (
                    <Ionicons
                      name={meta.icon}
                      size={15}
                      color={colors.text}
                      style={{ marginBottom: 6, opacity: 0.7 }}
                    />
                  ) : null}
                  <Text style={styles.bodyText}>{cleanBody(sec.body)}</Text>
                </View>
              </View>
            );
          })}

          <Pressable
            onPress={() => q.refetch()}
            disabled={q.isFetching}
            style={({ pressed }) => [
              styles.refreshBtn,
              pressed && { opacity: 0.85 },
              q.isFetching && { opacity: 0.6 },
            ]}
          >
            {q.isFetching ? (
              <ActivityIndicator color={colors.text} size="small" />
            ) : (
              <>
                <Ionicons name="refresh" size={15} color={colors.text} />
                <Text style={styles.refreshBtnText}>Re-run review</Text>
              </>
            )}
          </Pressable>
        </>
      ) : null}
    </Screen>
  );
}

const styles = StyleSheet.create({
  loading: {
    paddingVertical: spacing.xxl + spacing.lg,
    alignItems: 'center',
    gap: spacing.md,
  },
  loadingText: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 13,
  },
  proCard: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.xl,
    alignItems: 'center',
    gap: spacing.sm,
    ...glow.card,
  },
  proTitle: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 16,
    textAlign: 'center',
    marginTop: spacing.xs,
  },
  proSub: {
    color: colors.textDim,
    fontFamily: font.regular,
    fontSize: 13,
    lineHeight: 19,
    textAlign: 'center',
  },
  overviewCard: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    backgroundColor: colors.card,
    borderRadius: radius.lg,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    ...glow.card,
  },
  overviewText: {
    flex: 1,
    color: colors.text,
    fontFamily: font.medium,
    fontSize: 13,
  },
  section: { marginTop: spacing.xl },
  bodyCard: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1,
    borderColor: colors.border,
    padding: spacing.md,
    ...glow.card,
  },
  bodyText: {
    color: colors.text,
    fontFamily: font.regular,
    fontSize: 14,
    lineHeight: 20,
  },
  refreshBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    marginTop: spacing.xl,
    paddingVertical: spacing.md,
    borderRadius: radius.lg,
    backgroundColor: colors.bgElev,
    borderWidth: 1,
    borderColor: colors.border,
  },
  refreshBtnText: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14,
  },
});
