import { buildServiceConfig } from "./config/channel-config";
import type { PlatformConfig } from "./config";
import { ChannelRepository } from "./db/repositories/channels";
import { TopicsUsedRepository } from "./db/repositories/topics-used";
import { VideoRepository } from "./db/repositories/videos";
import { LlmService } from "./services/llm";
import { VideoService } from "./services/video";
import { VoiceService } from "./services/voice";
import { YouTubeService } from "./services/youtube";
import type { PipelineResult } from "./types/video";
import { estimatePipelineCostUsd } from "./utils/cost";
import { cleanupTmpDir, ensureTmpDir } from "./utils/tmp";

export interface RunPipelineOptions {
  channelId: string;
  topic?: string;
}

export class PipelineOrchestrator {
  private readonly platform: PlatformConfig;
  private readonly channels: ChannelRepository;
  private readonly videos: VideoRepository;
  private readonly topicsUsed: TopicsUsedRepository;

  constructor(platform: PlatformConfig) {
    this.platform = platform;
    this.channels = new ChannelRepository(platform.encryptionKey);
    this.videos = new VideoRepository();
    this.topicsUsed = new TopicsUsedRepository();
  }

  async run(options: RunPipelineOptions): Promise<PipelineResult> {
    const channel = await this.channels.findDecryptedById(options.channelId);
    if (!channel) {
      throw new Error(`Channel not found: ${options.channelId}`);
    }

    if (channel.status !== "active") {
      throw new Error(
        `Channel "${channel.name}" is ${channel.status}; pipeline aborted`,
      );
    }

    const monthlySpend = await this.channels.getMonthlySpend(channel.id);
    const monthlyBudget = Number(channel.monthly_budget_usd);
    const estimatedCost = estimatePipelineCostUsd();

    if (monthlyBudget > 0 && monthlySpend + estimatedCost > monthlyBudget) {
      throw new Error(
        `Monthly budget exceeded for channel "${channel.name}" (${monthlySpend.toFixed(2)} / ${monthlyBudget.toFixed(2)} USD)`,
      );
    }

    const serviceConfig = buildServiceConfig(this.platform, channel);
    const llm = new LlmService(serviceConfig);
    const voice = new VoiceService(serviceConfig);
    const video = new VideoService(serviceConfig);
    const youtube = new YouTubeService(serviceConfig);

    const excludedTopics = await this.topicsUsed.listTopicTexts(channel.id);
    const videoRecord = await this.videos.createProcessing(
      channel.id,
      options.topic ?? null,
    );

    const runDir = await ensureTmpDir(this.platform);
    console.log(
      `[pipeline] channel=${channel.name} (${channel.id}) run=${videoRecord.id} dir=${runDir}`,
    );

    try {
      console.log("[pipeline] generating script...");
      const payload = await llm.generateScript({
        topic: options.topic,
        excludedTopics,
      });
      console.log(
        `[pipeline] script ready: topic="${payload.topic}" title="${payload.title}"`,
      );

      console.log("[pipeline] synthesizing voiceover...");
      const voiceoverPath = await voice.synthesizeVoiceover(payload, runDir);
      console.log(`[pipeline] voiceover saved: ${voiceoverPath}`);

      console.log("[pipeline] rendering video...");
      const { renderedVideoUrl, localVideoPath } = await video.renderVideo(
        payload,
        voiceoverPath,
        runDir,
      );
      console.log(`[pipeline] video rendered: ${renderedVideoUrl}`);

      console.log("[pipeline] uploading to YouTube...");
      const upload = await youtube.uploadVideo(payload, localVideoPath);
      console.log(
        `[pipeline] upload complete (${upload.privacyStatus}): ${upload.videoUrl}`,
      );

      const costUsd = estimatePipelineCostUsd();
      await this.videos.markPrivate(videoRecord.id, {
        topic: payload.topic,
        title: payload.title,
        youtubeVideoId: upload.videoId,
        costUsd,
      });
      await this.topicsUsed.recordTopic(channel.id, payload.topic);

      return {
        videoId: upload.videoId,
        videoUrl: upload.videoUrl,
        title: payload.title,
        topic: payload.topic,
        privacyStatus: upload.privacyStatus,
        dbVideoId: videoRecord.id,
        costUsd,
      };
    } catch (error) {
      await this.videos.markFailed(videoRecord.id);
      throw error;
    } finally {
      await cleanupTmpDir(runDir);
      console.log(`[pipeline] cleaned up ${runDir}`);
    }
  }
}
