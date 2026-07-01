import type { AppConfig } from "./config";
import { LlmService } from "./services/llm";
import { VideoService } from "./services/video";
import { VoiceService } from "./services/voice";
import { YouTubeService } from "./services/youtube";
import type { PipelineResult } from "./types/video";
import { cleanupTmpDir, ensureTmpDir } from "./utils/tmp";

export interface RunPipelineOptions {
  topic?: string;
}

export class PipelineOrchestrator {
  private readonly config: AppConfig;
  private readonly llm: LlmService;
  private readonly voice: VoiceService;
  private readonly video: VideoService;
  private readonly youtube: YouTubeService;

  constructor(config: AppConfig) {
    this.config = config;
    this.llm = new LlmService(config);
    this.voice = new VoiceService(config);
    this.video = new VideoService(config);
    this.youtube = new YouTubeService(config);
  }

  async run(options: RunPipelineOptions = {}): Promise<PipelineResult> {
    const runDir = await ensureTmpDir(this.config);
    console.log(`[pipeline] started run in ${runDir}`);

    try {
      console.log("[pipeline] generating script...");
      const payload = await this.llm.generateScript(options.topic);
      console.log(`[pipeline] script ready: "${payload.title}"`);

      console.log("[pipeline] synthesizing voiceover...");
      const voiceoverPath = await this.voice.synthesizeVoiceover(
        payload,
        runDir,
      );
      console.log(`[pipeline] voiceover saved: ${voiceoverPath}`);

      console.log("[pipeline] rendering video...");
      const { renderedVideoUrl, localVideoPath } =
        await this.video.renderVideo(payload, voiceoverPath, runDir);
      console.log(`[pipeline] video rendered: ${renderedVideoUrl}`);

      console.log("[pipeline] uploading to YouTube...");
      const upload = await this.youtube.uploadVideo(payload, localVideoPath);
      console.log(
        `[pipeline] upload complete (${upload.privacyStatus}): ${upload.videoUrl}`,
      );

      return {
        videoId: upload.videoId,
        videoUrl: upload.videoUrl,
        title: payload.title,
        privacyStatus: upload.privacyStatus,
        voiceoverPath,
        renderedVideoUrl,
      };
    } finally {
      await cleanupTmpDir(runDir);
      console.log(`[pipeline] cleaned up ${runDir}`);
    }
  }
}
