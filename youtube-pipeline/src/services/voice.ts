import { createWriteStream } from "node:fs";
import { finished } from "node:stream/promises";
import path from "node:path";
import { Readable } from "node:stream";

import type { AppConfig } from "../config";
import type { VideoPayload } from "../types/video";
import { withRetry } from "../utils/tmp";

export class VoiceService {
  private readonly config: AppConfig;

  constructor(config: AppConfig) {
    this.config = config;
  }

  buildMasterScript(payload: VideoPayload): string {
    return payload.scenes
      .map((scene) => scene.voiceover_text.trim())
      .filter(Boolean)
      .join("\n\n");
  }

  async synthesizeVoiceover(
    payload: VideoPayload,
    runDir: string,
  ): Promise<string> {
    const masterScript = this.buildMasterScript(payload);
    const outputPath = path.join(runDir, "voiceover.mp3");

    await withRetry(
      async () => {
        const url = `https://api.elevenlabs.io/v1/text-to-speech/${this.config.elevenlabs.voiceId}`;
        const response = await fetch(url, {
          method: "POST",
          headers: {
            "xi-api-key": this.config.elevenlabs.apiKey,
            "Content-Type": "application/json",
            Accept: "audio/mpeg",
          },
          body: JSON.stringify({
            text: masterScript,
            model_id: this.config.elevenlabs.modelId,
            voice_settings: {
              stability: 0.45,
              similarity_boost: 0.8,
              style: 0.2,
              use_speaker_boost: true,
            },
          }),
        });

        if (!response.ok) {
          const errorBody = await response.text();
          throw new Error(
            `ElevenLabs TTS failed (${response.status}): ${errorBody}`,
          );
        }

        if (!response.body) {
          throw new Error("ElevenLabs TTS response had no body stream");
        }

        const nodeStream = Readable.fromWeb(
          response.body as import("node:stream/web").ReadableStream,
        );
        const fileStream = createWriteStream(outputPath);
        nodeStream.pipe(fileStream);
        await finished(fileStream);
      },
      {
        ...this.config.retry,
        label: "elevenlabs-tts",
      },
    );

    return outputPath;
  }
}
