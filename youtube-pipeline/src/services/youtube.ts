import { createReadStream } from "node:fs";
import { google, type youtube_v3 } from "googleapis";

import type { ServiceConfig } from "../config/channel-config";
import type { VideoPayload } from "../types/video";
import { withRetry } from "../utils/tmp";

export interface YouTubeUploadResult {
  videoId: string;
  videoUrl: string;
  privacyStatus: string;
}

export interface ChannelMonetizationStats {
  subsCount: number;
  watchHoursTotal: number;
  monetizationEligible: boolean;
}

export class YouTubeService {
  private readonly config: ServiceConfig;
  private readonly youtube: youtube_v3.Youtube;
  private readonly oauth2Client: InstanceType<typeof google.auth.OAuth2>;

  constructor(config: ServiceConfig) {
    this.config = config;

    this.oauth2Client = new google.auth.OAuth2(
      config.youtube.clientId,
      config.youtube.clientSecret,
    );

    this.oauth2Client.setCredentials({
      refresh_token: config.youtube.refreshToken,
    });

    this.youtube = google.youtube({
      version: "v3",
      auth: this.oauth2Client,
    });
  }

  async uploadVideo(
    payload: VideoPayload,
    localVideoPath: string,
  ): Promise<YouTubeUploadResult> {
    return withRetry(
      async () => {
        const privacyStatus = this.config.youtube.privacyStatus;

        const response = await this.youtube.videos.insert({
          part: ["snippet", "status"],
          requestBody: {
            snippet: {
              title: payload.title,
              description: payload.description,
              tags: payload.tags,
              categoryId: this.config.youtube.categoryId,
            },
            status: {
              privacyStatus,
              selfDeclaredMadeForKids: false,
            },
          },
          media: {
            body: createReadStream(localVideoPath),
          },
        });

        const videoId = response.data.id;
        if (!videoId) {
          throw new Error("YouTube upload succeeded but no video ID returned");
        }

        return {
          videoId,
          videoUrl: `https://www.youtube.com/watch?v=${videoId}`,
          privacyStatus,
        };
      },
      {
        ...this.config.retry,
        label: "youtube-upload",
      },
    );
  }

  async publishVideo(youtubeVideoId: string): Promise<void> {
    await withRetry(
      async () => {
        await this.youtube.videos.update({
          part: ["status"],
          requestBody: {
            id: youtubeVideoId,
            status: {
              privacyStatus: "public",
              selfDeclaredMadeForKids: false,
            },
          },
        });
      },
      {
        ...this.config.retry,
        label: "youtube-publish",
      },
    );
  }

  async fetchMonetizationStats(): Promise<ChannelMonetizationStats> {
    const channelResponse = await this.youtube.channels.list({
      part: ["statistics"],
      mine: true,
    });

    const channel = channelResponse.data.items?.[0];
    const subsCount = Number(channel?.statistics?.subscriberCount ?? 0);

    let watchHoursTotal = 0;

    try {
      const analytics = google.youtubeAnalytics({
        version: "v2",
        auth: this.oauth2Client,
      });

      const endDate = new Date();
      const startDate = new Date();
      startDate.setFullYear(startDate.getFullYear() - 1);

      const formatDate = (date: Date): string => date.toISOString().slice(0, 10);

      const analyticsResponse = await analytics.reports.query({
        ids: "channel==MINE",
        startDate: formatDate(startDate),
        endDate: formatDate(endDate),
        metrics: "estimatedMinutesWatched",
      });

      const minutesWatched = Number(
        analyticsResponse.data.rows?.[0]?.[0] ?? 0,
      );
      watchHoursTotal = minutesWatched / 60;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(
        `[youtube] analytics unavailable for channel ${this.config.channelId}: ${message}`,
      );
    }

    const monetizationEligible = subsCount >= 1000 && watchHoursTotal >= 4000;

    return {
      subsCount,
      watchHoursTotal,
      monetizationEligible,
    };
  }
}
