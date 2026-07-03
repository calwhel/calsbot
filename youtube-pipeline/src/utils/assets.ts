import { createReadStream, existsSync } from "node:fs";
import { stat } from "node:fs/promises";
import path from "node:path";
import { randomUUID } from "node:crypto";

interface RegisteredAsset {
  filePath: string;
  mimeType: string;
  expiresAt: number;
}

const assets = new Map<string, RegisteredAsset>();

const DEFAULT_TTL_MS = 30 * 60 * 1000;

export function registerTransientAsset(
  filePath: string,
  mimeType: string,
  ttlMs: number = DEFAULT_TTL_MS,
): string {
  const token = randomUUID();
  assets.set(token, {
    filePath,
    mimeType,
    expiresAt: Date.now() + ttlMs,
  });
  return token;
}

export function resolveTransientAsset(token: string): RegisteredAsset | null {
  const asset = assets.get(token);
  if (!asset) {
    return null;
  }

  if (Date.now() > asset.expiresAt) {
    assets.delete(token);
    return null;
  }

  if (!existsSync(asset.filePath)) {
    assets.delete(token);
    return null;
  }

  return asset;
}

export function revokeTransientAsset(token: string): void {
  assets.delete(token);
}

export function buildPublicAssetUrl(
  publicBaseUrl: string,
  token: string,
): string {
  const normalizedBase = publicBaseUrl.replace(/\/$/, "");
  return `${normalizedBase}/internal/assets/${token}`;
}

export async function assertReadableFile(filePath: string): Promise<void> {
  const fileStat = await stat(filePath);
  if (!fileStat.isFile() || fileStat.size === 0) {
    throw new Error(`Asset is missing or empty: ${filePath}`);
  }
}

export function streamTransientAsset(
  token: string,
): { stream: ReturnType<typeof createReadStream>; mimeType: string } | null {
  const asset = resolveTransientAsset(token);
  if (!asset) {
    return null;
  }

  return {
    stream: createReadStream(path.resolve(asset.filePath)),
    mimeType: asset.mimeType,
  };
}
