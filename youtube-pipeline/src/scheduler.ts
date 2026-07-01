import cron, { type ScheduledTask } from "node-cron";

import type { PlatformConfig } from "./config";
import { ChannelRepository } from "./db/repositories/channels";
import { bootstrapSchema } from "./db/pool";
import type { PipelineOrchestrator } from "./pipeline";
import { runPipelineForChannel } from "./routes/pipeline";

export class ChannelScheduler {
  private readonly channels: ChannelRepository;
  private readonly jobs = new Map<string, ScheduledTask>();
  private reloadTask: ScheduledTask | null = null;

  constructor(
    platform: PlatformConfig,
    private readonly orchestrator: PipelineOrchestrator,
  ) {
    this.channels = new ChannelRepository(platform.encryptionKey);
  }

  async start(): Promise<void> {
    await bootstrapSchema();
    await this.reloadJobs();

    this.reloadTask = cron.schedule("*/5 * * * *", () => {
      void this.reloadJobs();
    });

    console.log("[scheduler] started; reloading channel cron jobs every 5 minutes");
  }

  async stop(): Promise<void> {
    this.reloadTask?.stop();
    for (const job of this.jobs.values()) {
      job.stop();
    }
    this.jobs.clear();
  }

  private async reloadJobs(): Promise<void> {
    try {
      for (const job of this.jobs.values()) {
        job.stop();
      }
      this.jobs.clear();

      const activeChannels = await this.channels.listActive();

      for (const channel of activeChannels) {
        if (!cron.validate(channel.upload_frequency)) {
          console.error(
            `[scheduler] invalid cron for channel ${channel.name} (${channel.id}): ${channel.upload_frequency}`,
          );
          continue;
        }

        const job = cron.schedule(channel.upload_frequency, () => {
          console.log(
            `[scheduler] triggering pipeline for channel ${channel.name} (${channel.id})`,
          );
          void runPipelineForChannel(this.orchestrator, channel.id);
        });

        this.jobs.set(channel.id, job);
        console.log(
          `[scheduler] registered cron "${channel.upload_frequency}" for channel ${channel.name}`,
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error(`[scheduler] failed to reload jobs: ${message}`);
    }
  }
}
