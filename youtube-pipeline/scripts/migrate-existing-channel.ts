import path from "node:path";

import dotenv from "dotenv";

import { loadConfig } from "../src/config";
import { bootstrapSchema, closePool } from "../src/db/pool";
import { ChannelRepository } from "../src/db/repositories/channels";

dotenv.config({ path: path.join(__dirname, "..", ".env") });

function requireLegacyEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    throw new Error(
      `Missing ${name} in .env. This migration script imports your legacy single-channel config.`,
    );
  }
  return value;
}

async function main(): Promise<void> {
  const platform = loadConfig();
  await bootstrapSchema();

  const channels = new ChannelRepository(platform.encryptionKey);
  const existing = await channels.listAll();

  if (existing.length > 0) {
    console.log(
      `Channels table already has ${existing.length} row(s). Skipping migration.`,
    );
    console.log(
      "If you need to import another legacy channel, create it via POST /api/channels instead.",
    );
    await closePool();
    return;
  }

  const channel = await channels.create({
    name:
      process.env.MIGRATION_CHANNEL_NAME?.trim() || "Default Channel",
    niche_prompt:
      process.env.DEFAULT_TOPIC?.trim() ||
      process.env.MIGRATION_NICHE_PROMPT?.trim() ||
      "Deep-Dive Cosmic Mysteries",
    youtube_client_id: requireLegacyEnv("YOUTUBE_CLIENT_ID"),
    youtube_client_secret: requireLegacyEnv("YOUTUBE_CLIENT_SECRET"),
    youtube_refresh_token: requireLegacyEnv("YOUTUBE_REFRESH_TOKEN"),
    elevenlabs_voice_id: requireLegacyEnv("ELEVENLABS_VOICE_ID"),
    creatomate_template_id: requireLegacyEnv("CREATOMATE_TEMPLATE_ID"),
    upload_frequency:
      process.env.MIGRATION_UPLOAD_FREQUENCY?.trim() || "0 14 * * *",
    monthly_budget_usd: Number(process.env.MIGRATION_MONTHLY_BUDGET_USD ?? 100),
    status: "active",
  });

  console.log("Migration complete. Created initial channel:");
  console.log(JSON.stringify(channel, null, 2));
  console.log("");
  console.log(
    "Next steps:\n" +
      "1. Remove legacy per-channel env vars from Railway except shared platform keys.\n" +
      "2. Trigger the pipeline with POST /api/run-pipeline and body { \"channel_id\": \"...\" }.\n" +
      "3. Manage additional channels via the dashboard API.",
  );

  await closePool();
}

main().catch(async (error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Migration failed: ${message}`);
  await closePool();
  process.exit(1);
});
