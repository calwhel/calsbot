import type { DecryptedChannel } from "../db/repositories/channels";
import type { PlatformConfig } from "./index";

export interface ServiceConfig {
  port: number;
  authToken: string;
  tmpDir: string;
  anthropic: PlatformConfig["anthropic"];
  elevenlabs: PlatformConfig["elevenlabs"] & { voiceId: string };
  creatomate: PlatformConfig["creatomate"] & { templateId: string };
  publicBaseUrl: string;
  youtube: {
    clientId: string;
    clientSecret: string;
    refreshToken: string;
    privacyStatus: "private" | "unlisted";
    categoryId: string;
  };
  retry: PlatformConfig["retry"];
  channelId: string;
  nichePrompt: string;
}

export function buildServiceConfig(
  platform: PlatformConfig,
  channel: DecryptedChannel,
): ServiceConfig {
  return {
    port: platform.port,
    authToken: platform.authToken,
    tmpDir: platform.tmpDir,
    anthropic: platform.anthropic,
    elevenlabs: {
      ...platform.elevenlabs,
      voiceId: channel.elevenlabs_voice_id,
    },
    creatomate: {
      ...platform.creatomate,
      templateId: channel.creatomate_template_id,
    },
    publicBaseUrl: platform.publicBaseUrl,
    youtube: {
      clientId: channel.youtube_client_id,
      clientSecret: channel.youtube_client_secret,
      refreshToken: channel.youtube_refresh_token,
      privacyStatus: platform.youtube.privacyStatus,
      categoryId: platform.youtube.categoryId,
    },
    retry: platform.retry,
    channelId: channel.id,
    nichePrompt: channel.niche_prompt,
  };
}
