import React from 'react';
import { View, Text, StyleSheet, ScrollView, ActivityIndicator, Pressable } from 'react-native';
import { useLocalSearchParams, Stack, useRouter } from 'expo-router';
import { useQuery } from '@tanstack/react-query';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';

import { EmptyState } from '@/components/EmptyState';
import { Pill } from '@/components/Pill';
import { colors, font, glow, radius, spacing } from '@/constants/colors';
import { useAuth } from '@/contexts/AuthContext';
import { apiGet, type CreatorProfile } from '@/lib/api';

function fmtPnl(v: number | null | undefined): string {
  if (v == null) return '—';
  const sign = v > 0 ? '+' : '';
  return `${sign}${v.toFixed(2)}%`;
}

function paletteFor(name: string): [string, string] {
  const palettes: Array<[string, string]> = [
    ['#a78bfa', '#7c3aed'],
    ['#22d3ee', '#3b82f6'],
    ['#34d399', '#10b981'],
    ['#fbbf24', '#f59e0b'],
    ['#f472b6', '#db2777'],
  ];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return palettes[h % palettes.length];
}

export default function CreatorProfileScreen() {
  const { uid: creatorUid } = useLocalSearchParams<{ uid: string }>();
  const insets = useSafeAreaInsets();
  const { uid } = useAuth();
  const router = useRouter();

  const profileQ = useQuery({
    queryKey: ['creator-profile', creatorUid, uid],
    queryFn: () => apiGet<CreatorProfile>(`/api/creator/${creatorUid}`, uid),
    enabled: !!uid && !!creatorUid,
  });

  if (profileQ.isLoading) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <View style={styles.center}>
          <ActivityIndicator color={colors.accent} size="large" />
        </View>
      </>
    );
  }

  if (profileQ.isError || !profileQ.data) {
    return (
      <>
        <Stack.Screen options={{ title: '' }} />
        <EmptyState
          icon="person-remove-outline"
          title="Creator not found"
          hint="They may have removed their account."
        />
      </>
    );
  }

  const p = profileQ.data;
  const initial = (p.name || '?').charAt(0).toUpperCase();
  const [c0, c1] = paletteFor(p.name || '?');

  return (
    <>
      <Stack.Screen options={{ title: p.name }} />
      <ScrollView
        style={{ flex: 1, backgroundColor: colors.bg }}
        contentContainerStyle={[styles.content, { paddingBottom: insets.bottom + 32 }]}
        showsVerticalScrollIndicator={false}
      >
        {/* Profile header */}
        <View style={styles.header}>
          <View style={[styles.avatar, { backgroundColor: c0 }]}>
            <View style={[styles.avatarInner, { backgroundColor: c1 }]}>
              <Text style={styles.avatarText}>{initial}</Text>
            </View>
          </View>
          <Text style={styles.name}>{p.name}</Text>
          <Text style={styles.joined}>Member since {p.joined}</Text>

          <View style={styles.statsRow}>
            <View style={styles.statCell}>
              <Text style={styles.statValue}>{p.strategy_count}</Text>
              <Text style={styles.statLabel}>Strategies</Text>
            </View>
            <View style={styles.statDiv} />
            <View style={styles.statCell}>
              <Text style={styles.statValue}>{p.follower_count}</Text>
              <Text style={styles.statLabel}>Followers</Text>
            </View>
            <View style={styles.statDiv} />
            <View style={styles.statCell}>
              <Text style={styles.statValue}>{p.total_subscribers}</Text>
              <Text style={styles.statLabel}>Subs</Text>
            </View>
          </View>
        </View>

        {/* Strategy list */}
        <Text style={styles.sectionLabel}>Published strategies</Text>
        {p.strategies.length === 0 ? (
          <EmptyState
            icon="albums-outline"
            title="No strategies yet"
            hint="This creator hasn't published anything to the marketplace."
          />
        ) : (
          <View style={styles.list}>
            {p.strategies.map((s, i) => {
              const pnl = s.total_pnl ?? null;
              const wr  = s.win_rate ?? null;
              const trades = s.total_trades ?? 0;
              const isFree = (s.pricing_model || 'free') === 'free';
              return (
                <Pressable
                  key={`s-${s.id}`}
                  onPress={() => router.push(`/listing/${s.id}` as any)}
                  style={({ pressed }) => [
                    styles.listRow,
                    i > 0 && styles.listRowDiv,
                    pressed && { opacity: 0.85 },
                  ]}
                >
                  <View style={{ flex: 1 }}>
                    <Text style={styles.listTitle} numberOfLines={2}>{s.title}</Text>
                    {s.summary ? (
                      <Text style={styles.listSummary} numberOfLines={2}>{s.summary}</Text>
                    ) : null}
                    <View style={styles.listMeta}>
                      {s.is_verified ? <Pill label="✓ Verified" tone="accent" small /> : null}
                      <Pill
                        label={isFree ? 'FREE' : `$${(s.price_usdt ?? 0).toFixed(0)}`}
                        tone={isFree ? 'positive' : 'warning'}
                        small
                      />
                      {(s.clone_count ?? 0) > 0 ? (
                        <View style={styles.metaPill}>
                          <Ionicons name="copy-outline" size={10} color={colors.textMute} />
                          <Text style={styles.metaPillText}>{s.clone_count}</Text>
                        </View>
                      ) : null}
                    </View>
                  </View>
                  <View style={styles.listStats}>
                    {trades > 0 && pnl != null ? (
                      <>
                        <Text style={[styles.listPnl, { color: pnl > 0 ? colors.positive : pnl < 0 ? colors.negative : colors.text }]}>
                          {fmtPnl(pnl)}
                        </Text>
                        <Text style={styles.listWr}>
                          {wr != null ? `${wr.toFixed(0)}% · ${trades}t` : `${trades}t`}
                        </Text>
                      </>
                    ) : (
                      <>
                        <Text style={[styles.listPnl, { color: colors.textMute }]}>—</Text>
                        <Text style={styles.listWr}>No trades</Text>
                      </>
                    )}
                  </View>
                </Pressable>
              );
            })}
          </View>
        )}
      </ScrollView>
    </>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: spacing.lg, paddingTop: spacing.xl },
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.bg },
  header: {
    alignItems: 'center',
    paddingVertical: spacing.lg,
  },
  avatar: {
    width: 76, height: 76, borderRadius: 38,
    alignItems: 'center', justifyContent: 'center',
    padding: 4,
  },
  avatarInner: {
    width: 68, height: 68, borderRadius: 34,
    alignItems: 'center', justifyContent: 'center',
  },
  avatarText: {
    color: '#fff',
    fontFamily: font.black,
    fontSize: 28,
    letterSpacing: -1,
  },
  name: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 22,
    letterSpacing: -0.5,
    marginTop: spacing.md,
  },
  joined: {
    color: colors.textDim,
    fontFamily: font.medium,
    fontSize: 12.5,
    marginTop: 2,
  },
  statsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing.lg,
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1, borderColor: colors.border,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    alignSelf: 'stretch',
    ...glow.card,
  },
  statCell: { flex: 1, alignItems: 'center' },
  statDiv: { width: 1, height: 26, backgroundColor: colors.divider },
  statValue: {
    color: colors.text,
    fontFamily: font.black,
    fontSize: 18,
    fontVariant: ['tabular-nums'],
    letterSpacing: -0.4,
  },
  statLabel: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
    letterSpacing: 0.4,
    textTransform: 'uppercase',
    marginTop: 2,
  },
  sectionLabel: {
    color: colors.textDim,
    fontFamily: font.bold,
    fontSize: 12,
    letterSpacing: 0.6,
    textTransform: 'uppercase',
    marginTop: spacing.xl,
    marginBottom: spacing.sm,
  },
  list: {
    backgroundColor: colors.card,
    borderRadius: radius.xl,
    borderWidth: 1, borderColor: colors.border,
    overflow: 'hidden',
    ...glow.card,
  },
  listRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
    padding: spacing.md,
  },
  listRowDiv: { borderTopWidth: 1, borderTopColor: colors.divider },
  listTitle: {
    color: colors.text,
    fontFamily: font.bold,
    fontSize: 14.5,
    letterSpacing: -0.2,
  },
  listSummary: {
    color: colors.textMute,
    fontFamily: font.regular,
    fontSize: 12,
    lineHeight: 16,
    marginTop: 3,
  },
  listMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 5,
    marginTop: 6,
    flexWrap: 'wrap',
  },
  metaPill: {
    flexDirection: 'row', alignItems: 'center', gap: 3,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: radius.pill,
    backgroundColor: colors.bgElev,
    borderWidth: 1, borderColor: colors.border,
  },
  metaPillText: {
    color: colors.textMute,
    fontFamily: font.bold,
    fontSize: 10,
  },
  listStats: { alignItems: 'flex-end' },
  listPnl: {
    fontFamily: font.black,
    fontSize: 14,
    fontVariant: ['tabular-nums'],
    letterSpacing: -0.2,
  },
  listWr: {
    color: colors.textMute,
    fontFamily: font.medium,
    fontSize: 10.5,
    marginTop: 2,
  },
});
