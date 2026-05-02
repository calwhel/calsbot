import React, { createContext, useCallback, useContext, useEffect, useState } from 'react';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';
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
  signOut: () => Promise<void>;
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

  const signOut = useCallback(async () => {
    await storage.del(UID_STORE_KEY);
    setUser(null);
    setUid(null);
  }, []);

  return (
    <AuthCtx.Provider value={{ user, uid, ready, signIn, signOut }}>
      {children}
    </AuthCtx.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthCtx);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
