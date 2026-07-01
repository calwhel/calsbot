import type { Request, Response, Router } from "express";

import { buildServiceConfig } from "../config/channel-config";
import type { PlatformConfig } from "../config";
import { ChannelRepository } from "../db/repositories/channels";
import { VideoRepository } from "../db/repositories/videos";
import type { PipelineOrchestrator } from "../pipeline";
import { YouTubeService } from "../services/youtube";

const runningChannels = new Set<string>();

export function isChannelPipelineRunning(channelId: string): boolean {
  return runningChannels.has(channelId);
}

export function getRunningChannelCount(): number {
  return runningChannels.size;
}

export function createPipelineRoutes(
  router: Router,
  platform: PlatformConfig,
  orchestrator: PipelineOrchestrator,
): void {
  const handleRunPipeline = async (req: Request, res: Response): Promise<void> => {
    const channelId =
      typeof req.body?.channel_id === "string"
        ? req.body.channel_id
        : typeof req.query.channel_id === "string"
          ? req.query.channel_id
          : undefined;

    if (!channelId) {
      res.status(400).json({ error: "channel_id is required" });
      return;
    }

    if (runningChannels.has(channelId)) {
      res.status(409).json({ error: "Pipeline is already running for this channel" });
      return;
    }

    runningChannels.add(channelId);
    const topic =
      typeof req.body?.topic === "string" ? req.body.topic : undefined;

    try {
      const result = await orchestrator.run({ channelId, topic });
      res.status(200).json({ success: true, result });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[pipeline] channel=${channelId} failed:`, message);
      res.status(500).json({ success: false, error: message });
    } finally {
      runningChannels.delete(channelId);
    }
  };

  router.post("/run-pipeline", (req, res) => {
    void handleRunPipeline(req, res);
  });

  router.get("/run-pipeline", (req, res) => {
    void handleRunPipeline(req, res);
  });

  const videos = new VideoRepository();
  const channels = new ChannelRepository(platform.encryptionKey);

  router.get("/pending", async (_req, res) => {
    const pending = await videos.listPending();
    res.status(200).json({ videos: pending });
  });

  router.post("/publish/:video_id", async (req, res) => {
    const video = await videos.findById(req.params.video_id);
    if (!video) {
      res.status(404).json({ error: "Video not found" });
      return;
    }

    if (video.status !== "private" || !video.youtube_video_id) {
      res.status(400).json({
        error: "Only private uploaded videos can be published",
      });
      return;
    }

    const channel = await channels.findDecryptedById(video.channel_id);
    if (!channel) {
      res.status(404).json({ error: "Channel not found for video" });
      return;
    }

    try {
      const serviceConfig = buildServiceConfig(platform, channel);
      const youtube = new YouTubeService(serviceConfig);
      await youtube.publishVideo(video.youtube_video_id);
      const updated = await videos.markPublished(video.id);

      res.status(200).json({
        success: true,
        video: updated,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(500).json({ success: false, error: message });
    }
  });

  router.get("/costs", async (_req, res) => {
    const costs = await videos.getCostSummary();
    res.status(200).json({ costs });
  });

  router.get("/monetization", async (_req, res) => {
    const channelList = await channels.listAll();
    const results = [];

    for (const channel of channelList) {
      const decrypted = await channels.findDecryptedById(channel.id);
      if (!decrypted) {
        continue;
      }

      try {
        const serviceConfig = buildServiceConfig(platform, decrypted);
        const youtube = new YouTubeService(serviceConfig);
        const stats = await youtube.fetchMonetizationStats();

        await channels.upsertStats(channel.id, {
          subs_count: stats.subsCount,
          watch_hours_total: stats.watchHoursTotal,
          monetization_eligible: stats.monetizationEligible,
        });

        results.push({
          channel_id: channel.id,
          channel_name: channel.name,
          subs_count: stats.subsCount,
          watch_hours_total: stats.watchHoursTotal,
          monetization_eligible: stats.monetizationEligible,
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        results.push({
          channel_id: channel.id,
          channel_name: channel.name,
          error: message,
          stats: channel.stats,
        });
      }
    }

    res.status(200).json({ channels: results });
  });
}

export async function runPipelineForChannel(
  orchestrator: PipelineOrchestrator,
  channelId: string,
): Promise<void> {
  if (runningChannels.has(channelId)) {
    console.warn(
      `[scheduler] skipped channel ${channelId}; pipeline already running`,
    );
    return;
  }

  runningChannels.add(channelId);

  try {
    await orchestrator.run({ channelId });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[scheduler] channel ${channelId} failed:`, message);
  } finally {
    runningChannels.delete(channelId);
  }
}
