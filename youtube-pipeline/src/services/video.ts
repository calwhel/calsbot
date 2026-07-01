import path from "node:path";

import type { AppConfig } from "../config";
import type { VideoPayload } from "../types/video";
import {
  assertReadableFile,
  buildPublicAssetUrl,
  registerTransientAsset,
  revokeTransientAsset,
} from "../utils/assets";
import { sleep, withRetry } from "../utils/tmp";

interface CreatomateRenderResponse {
  id: string;
  status: string;
  url?: string;
  error_message?: string;
}

export class VideoService {
  private readonly config: AppConfig;

  constructor(config: AppConfig) {
    this.config = config;
  }

  private buildModifications(
    payload: VideoPayload,
    audioSource: string,
  ): Record<string, unknown> {
    const modifications: Record<string, unknown> = {
      "Master-Audio.source": audioSource,
      "Title-Text.text": payload.title,
      "Thumbnail-Text.text": payload.thumbnail_text,
      "Thumbnail-Image.prompt": payload.thumbnail_prompt,
    };

    payload.scenes.forEach((scene, index) => {
      const sceneNumber = index + 1;
      modifications[`Scene-${sceneNumber}-Image.prompt`] = scene.visual_prompt;
      modifications[`Scene-${sceneNumber}-Overlay.text`] = scene.overlay_text;
      modifications[`Scene-${sceneNumber}-Voiceover.text`] =
        scene.voiceover_text;
    });

    return modifications;
  }

  private async createPublicAudioUrl(voiceoverPath: string): Promise<{
    audioSource: string;
    assetToken: string;
  }> {
    await assertReadableFile(voiceoverPath);
    const assetToken = registerTransientAsset(voiceoverPath, "audio/mpeg");
    const audioSource = buildPublicAssetUrl(
      this.config.publicBaseUrl,
      assetToken,
    );
    return { audioSource, assetToken };
  }

  private async createRender(
    payload: VideoPayload,
    audioSource: string,
  ): Promise<string> {
    const response = await fetch("https://api.creatomate.com/v2/renders", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.config.creatomate.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        template_id: this.config.creatomate.templateId,
        modifications: this.buildModifications(payload, audioSource),
      }),
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(
        `Creatomate render request failed (${response.status}): ${errorBody}`,
      );
    }

    const renders = (await response.json()) as CreatomateRenderResponse[];
    const render = Array.isArray(renders) ? renders[0] : renders;
    if (!render?.id) {
      throw new Error("Creatomate render response missing render id");
    }

    return render.id;
  }

  private async pollRender(renderId: string): Promise<string> {
    const startedAt = Date.now();

    while (Date.now() - startedAt < this.config.creatomate.pollTimeoutMs) {
      const response = await fetch(
        `https://api.creatomate.com/v1/renders/${renderId}`,
        {
          headers: {
            Authorization: `Bearer ${this.config.creatomate.apiKey}`,
          },
        },
      );

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(
          `Creatomate poll failed (${response.status}): ${errorBody}`,
        );
      }

      const render = (await response.json()) as CreatomateRenderResponse;
      if (render.status === "succeeded" && render.url) {
        return render.url;
      }

      if (render.status === "failed") {
        throw new Error(
          `Creatomate render failed: ${render.error_message ?? "unknown error"}`,
        );
      }

      await sleep(this.config.creatomate.pollIntervalMs);
    }

    throw new Error(
      `Creatomate render timed out after ${this.config.creatomate.pollTimeoutMs}ms`,
    );
  }

  async renderVideo(
    payload: VideoPayload,
    voiceoverPath: string,
    runDir: string,
  ): Promise<{ renderedVideoUrl: string; localVideoPath: string }> {
    return withRetry(
      async () => {
        const { audioSource, assetToken } =
          await this.createPublicAudioUrl(voiceoverPath);

        try {
          const renderId = await this.createRender(payload, audioSource);
          const renderedVideoUrl = await this.pollRender(renderId);
          const localVideoPath = path.join(runDir, "rendered-video.mp4");

          const videoResponse = await fetch(renderedVideoUrl);
          if (!videoResponse.ok) {
            throw new Error(
              `Failed to download rendered video (${videoResponse.status})`,
            );
          }

          const videoBuffer = Buffer.from(await videoResponse.arrayBuffer());
          const { writeFile } = await import("node:fs/promises");
          await writeFile(localVideoPath, videoBuffer);

          return { renderedVideoUrl, localVideoPath };
        } finally {
          revokeTransientAsset(assetToken);
        }
      },
      {
        ...this.config.retry,
        label: "creatomate-render",
      },
    );
  }
}
