import { createReadStream } from "node:fs";
import { google, type youtube_v3 } from "googleapis";

import type { AppConfig } from "../config";
import type { VideoPayload } from "../types/video";
import { withRetry } from "../utils/tmp";

export interface YouTubeUploadResult {
  videoId: string;
  videoUrl: string;
  privacyStatus: string;
}

export class YouTubeService {
  private readonly config: AppConfig;
  private readonly youtube: youtube_v3.Youtube;

  constructor(config: AppConfig) {
    this.config = config;

    const oauth2Client = new google.auth.OAuth2(
      config.youtube.clientId,
      config.youtube.clientSecret,
    );

    oauth2Client.setCredentials({
      refresh_token: config.youtube.refreshToken,
    });

    this.youtube = google.youtube({
      version: "v3",
      auth: oauth2Client,
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
}
