import { randomUUID } from "node:crypto";
import { mkdir, rm } from "node:fs/promises";
import path from "node:path";

import type { PlatformConfig } from "../config";

export async function ensureTmpDir(config: Pick<PlatformConfig, "tmpDir">): Promise<string> {
  const runDir = path.join(config.tmpDir, `pipeline-${randomUUID()}`);
  await mkdir(runDir, { recursive: true });
  return runDir;
}

export async function cleanupTmpDir(runDir: string): Promise<void> {
  await rm(runDir, { recursive: true, force: true });
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export interface RetryOptions {
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  label?: string;
}

export async function withRetry<T>(
  operation: () => Promise<T>,
  options: RetryOptions,
): Promise<T> {
  const label = options.label ?? "operation";
  let lastError: unknown;

  for (let attempt = 1; attempt <= options.maxAttempts; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt >= options.maxAttempts) {
        break;
      }

      const exponentialDelay = options.baseDelayMs * 2 ** (attempt - 1);
      const delay = Math.min(exponentialDelay, options.maxDelayMs);
      const message =
        error instanceof Error ? error.message : String(error);
      console.warn(
        `[retry] ${label} failed (attempt ${attempt}/${options.maxAttempts}): ${message}. Retrying in ${delay}ms...`,
      );
      await sleep(delay);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(`[retry] ${label} failed after ${options.maxAttempts} attempts`);
}

export async function downloadToFile(
  url: string,
  destinationPath: string,
): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to download asset (${response.status}): ${url}`,
    );
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  const { writeFile } = await import("node:fs/promises");
  await writeFile(destinationPath, buffer);
}
