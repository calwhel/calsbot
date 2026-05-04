import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';
import * as SecureStore from 'expo-secure-store';
import { AppState, Platform, type AppStateStatus } from 'react-native';
import { apiPost, ApiError, type LoginResponse } from '@/lib/api';

const UID_STORE_KEY = 'tradehub_uid';

// SecureStore is unavailable on web — fall back to localStorage there.
const storage = {
  async get(key: string): Promise<string | null> {
    if (Platform.OS === 'web') {
      try { return globalThis.localStorage?.getItem(key) ?? null; } catch { return null; }
    }
    return SecureStore.getItemAsync(key);
  },
  async set(key: string, value: string): Promise<void> {
    if (Platform.OS === 'web') {
      try { globalThis.localStorage?.setItem(key, value); } catch {}
      return;
    }
    return SecureStore.setItemAsync(key, value);
  },
  async del(key: string): Promise<void> {
    if (Platform.OS === 'web') {
      try { globalThis.localStorage?.removeItem(key); } catch {}
      return;
    }
    return SecureStore.deleteItemAsync(key);
  },
};

export type AuthState = {
  user: LoginResponse | null;
  uid: string | null;
  ready: boolean;
  signIn: (uid: string) => Promise<LoginResponse>;
  signInEmail: (email: string, password: string) => Promise<LoginResponse>;
  signInApple: (identityToken: string, fullName?: string | null, email?: string | null) => Promise<LoginResponse>;
  signOut: () => Promise<void>;
  refreshUser: () => Promise<LoginResponse | null>;
};

const AuthCtx = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<LoginResponse | null>(null);
  const [uid, setUid] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  // Restore + revalidate persisted UID on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stored = await storage.get(UID_STORE_KEY);
        if (stored) {
          try {
            const fresh = await apiPost<LoginResponse>('/api/mobile/login', { uid: stored });
            if (!cancelled) {
              setUser(fresh);
              setUid(fresh.uid);
            }
          } catch (e) {
            // Stored UID is invalid (e.g. wiped account) — clear it.
            if (e instanceof ApiError && e.status === 403) {
              await storage.del(UID_STORE_KEY);
            }
          }
        }
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const signIn = useCallback(async (rawUid: string): Promise<LoginResponse> => {
    const cleaned = rawUid.trim().toUpperCase();
    const fresh = await apiPost<LoginResponse>('/api/mobile/login', { uid: cleaned });
    await storage.set(UID_STORE_KEY, fresh.uid);
    setUser(fresh);
    setUid(fresh.uid);
    return fresh;
  }, []);

  const signInEmail = useCallback(async (email: string, password: string): Promise<LoginResponse> => {
    const fresh = await apiPost<LoginResponse>('/api/mobile/login/email', {
      email: email.trim().toLowerCase(),
      password,
    });
    await storage.set(UID_STORE_KEY, fresh.uid);
    setUser(fresh);
    setUid(fresh.uid);
    return fresh;
  }, []);

  const signInApple = useCallback(async (
    identityToken: string,
    fullName?: string | null,
    email?: string | null,
  ): Promise<LoginResponse> => {
    const fresh = await apiPost<LoginResponse>('/api/mobile/login/apple', {
      identity_token: identityToken,
      full_name: fullName || undefined,
      email: email || undefined,
    });
    await storage.set(UID_STORE_KEY, fresh.uid);
    setUser(fresh);
    setUid(fresh.uid);
    return fresh;
  }, []);

  const refreshUser = useCallback(async (): Promise<LoginResponse | null> => {
    const stored = uid || (await storage.get(UID_STORE_KEY));
    if (!stored) return null;
    try {
      const fresh = await apiPost<LoginResponse>('/api/mobile/login', { uid: stored });
      // Cancellation guard: if the user signed out (or signed in as a
      // different account) while this request was in flight, drop the result
      // so we don't accidentally restore the previous session.
      const currentStored = await storage.get(UID_STORE_KEY);
      if (currentStored !== fresh.uid) return null;
      setUser(fresh);
      setUid(fresh.uid);
      return fresh;
    } catch (e) {
      // Stored UID is invalid (e.g. wiped account) — clear it. Same
      // cancellation guard applies: only clear if the user hasn't already
      // signed out themselves.
      if (e instanceof ApiError && e.status === 403) {
        const currentStored = await storage.get(UID_STORE_KEY);
        if (currentStored === stored) {
          await storage.del(UID_STORE_KEY);
          setUser(null);
          setUid(null);
        }
      }
      return null;
    }
  }, [uid]);

  // Auto-refresh user payload when the app returns to the foreground. This
  // catches cases like "user upgraded to Pro on the web" so the mobile UI
  // reflects the new entitlement without a full sign-out / sign-in cycle.
  // We throttle to at most once every 15s to avoid hammering /api/mobile/login
  // on rapid background→foreground toggles.
  const lastRefreshRef = useRef<number>(0);
  useEffect(() => {
    const sub = AppState.addEventListener('change', (next: AppStateStatus) => {
      if (next !== 'active') return;
      if (!uid) return;
      const now = Date.now();
      if (now - lastRefreshRef.current < 15_000) return;
      lastRefreshRef.current = now;
      refreshUser().catch(() => { /* non-fatal */ });
    });
    return () => sub.remove();
  }, [uid, refreshUser]);

  const signOut = useCallback(async () => {
    // Best-effort: tell the backend to drop our push token so this device
    // stops receiving notifications for the signed-out user.
    try {
      const { unregisterPushToken } = await import('@/lib/notifications');
      await unregisterPushToken();
    } catch {
      // ignore
    }
    await storage.del(UID_STORE_KEY);
    setUser(null);
    setUid(null);
  }, []);

  // Whenever we have an authenticated UID, register for push notifications.
  // Idempotent — the helper caches the last registered token in-memory.
  useEffect(() => {
    if (!uid) return;
    let cancelled = false;
    (async () => {
      try {
        const { registerPushTokenForUid } = await import('@/lib/notifications');
        if (!cancelled) await registerPushTokenForUid(uid);
      } catch {
        // notifications module failed to load — non-fatal
      }
    })();
    return () => { cancelled = true; };
  }, [uid]);

  return (
    <AuthCtx.Provider value={{ user, uid, ready, signIn, signInEmail, signInApple, signOut, refreshUser }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
