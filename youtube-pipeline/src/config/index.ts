export interface PlatformConfig {
  port: number;
  authToken: string;
  tmpDir: string;
  encryptionKey: string;
  anthropic: {
    apiKey: string;
    model: string;
  };
  elevenlabs: {
    apiKey: string;
    modelId: string;
  };
  creatomate: {
    apiKey: string;
    pollIntervalMs: number;
    pollTimeoutMs: number;
  };
  publicBaseUrl: string;
  youtube: {
    privacyStatus: "private" | "unlisted";
    categoryId: string;
  };
  retry: {
    maxAttempts: number;
    baseDelayMs: number;
    maxDelayMs: number;
  };
}

function requireEnv(name: string): string {
  const value = process.env[name];
  if (!value || value.trim() === "") {
    throw new Error(`Missing required environment variable: ${name}`);
  }
  return value.trim();
}

function optionalEnv(name: string, fallback: string): string {
  const value = process.env[name];
  return value && value.trim() !== "" ? value.trim() : fallback;
}

function parsePositiveInt(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function resolvePublicBaseUrl(port: number): string {
  const explicit = process.env.PUBLIC_BASE_URL?.trim();
  if (explicit) {
    return explicit.replace(/\/$/, "");
  }

  const railwayDomain = process.env.RAILWAY_PUBLIC_DOMAIN?.trim();
  if (railwayDomain) {
    if (
      railwayDomain.startsWith("http://") ||
      railwayDomain.startsWith("https://")
    ) {
      return railwayDomain.replace(/\/$/, "");
    }
    return `https://${railwayDomain.replace(/\/$/, "")}`;
  }

  return `http://localhost:${port}`;
}

function parsePrivacyStatus(value: string): "private" | "unlisted" {
  const normalized = value.toLowerCase();
  if (normalized === "private" || normalized === "unlisted") {
    return normalized;
  }
  throw new Error(
    `YOUTUBE_PRIVACY_STATUS must be "private" or "unlisted", got: ${value}`,
  );
}

let cachedConfig: PlatformConfig | null = null;

export function loadConfig(): PlatformConfig {
  if (cachedConfig) {
    return cachedConfig;
  }

  cachedConfig = {
    port: parsePositiveInt(optionalEnv("PORT", "3000"), 3000),
    authToken: requireEnv("AUTH_TOKEN"),
    tmpDir: optionalEnv("TMP_DIR", "/tmp"),
    encryptionKey: requireEnv("ENCRYPTION_KEY"),
    anthropic: {
      apiKey: requireEnv("ANTHROPIC_API_KEY"),
      model: optionalEnv(
        "ANTHROPIC_MODEL",
        "claude-3-5-sonnet-20241022",
      ),
    },
    elevenlabs: {
      apiKey: requireEnv("ELEVENLABS_API_KEY"),
      modelId: optionalEnv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2"),
    },
    creatomate: {
      apiKey: requireEnv("CREATOMATE_API_KEY"),
      pollIntervalMs: parsePositiveInt(
        optionalEnv("CREATOMATE_POLL_INTERVAL_MS", "5000"),
        5000,
      ),
      pollTimeoutMs: parsePositiveInt(
        optionalEnv("CREATOMATE_POLL_TIMEOUT_MS", "900000"),
        900_000,
      ),
    },
    publicBaseUrl: resolvePublicBaseUrl(
      parsePositiveInt(optionalEnv("PORT", "3000"), 3000),
    ),
    youtube: {
      privacyStatus: parsePrivacyStatus(
        optionalEnv("YOUTUBE_PRIVACY_STATUS", "private"),
      ),
      categoryId: optionalEnv("YOUTUBE_CATEGORY_ID", "28"),
    },
    retry: {
      maxAttempts: parsePositiveInt(optionalEnv("RETRY_MAX_ATTEMPTS", "5"), 5),
      baseDelayMs: parsePositiveInt(
        optionalEnv("RETRY_BASE_DELAY_MS", "1000"),
        1000,
      ),
      maxDelayMs: parsePositiveInt(
        optionalEnv("RETRY_MAX_DELAY_MS", "30000"),
        30_000,
      ),
    },
  };

  return cachedConfig;
}

export function resetConfigForTests(): void {
  cachedConfig = null;
}

/** @deprecated Use PlatformConfig — kept for gradual migration in services */
export type AppConfig = import("./channel-config").ServiceConfig;
