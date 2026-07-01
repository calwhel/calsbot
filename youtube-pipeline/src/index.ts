import express from "express";

import { loadConfig } from "./config";
import { createAuthMiddleware } from "./middleware/auth";
import { PipelineOrchestrator } from "./pipeline";
import { createChannelRoutes } from "./routes/channels";
import {
  createPipelineRoutes,
  getRunningChannelCount,
} from "./routes/pipeline";
import { ChannelScheduler } from "./scheduler";
import { streamTransientAsset } from "./utils/assets";

async function main(): Promise<void> {
  const platform = loadConfig();
  const app = express();
  const orchestrator = new PipelineOrchestrator(platform);
  const scheduler = new ChannelScheduler(platform, orchestrator);
  const authenticate = createAuthMiddleware(platform);

  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_req, res) => {
    res.status(200).json({
      status: "ok",
      runningPipelines: getRunningChannelCount(),
      timestamp: new Date().toISOString(),
    });
  });

  app.get("/internal/assets/:token", (req, res) => {
    const asset = streamTransientAsset(req.params.token);
    if (!asset) {
      res.status(404).json({ error: "Asset not found or expired" });
      return;
    }

    res.setHeader("Content-Type", asset.mimeType);
    res.setHeader("Cache-Control", "no-store");
    asset.stream.pipe(res);
  });

  const apiRouter = express.Router();
  createChannelRoutes(apiRouter, platform);
  createPipelineRoutes(apiRouter, platform, orchestrator);
  app.use("/api", authenticate, apiRouter);

  await scheduler.start();

  app.listen(platform.port, () => {
    console.log(
      `YouTube pipeline server listening on port ${platform.port}`,
    );
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Fatal startup error: ${message}`);
  process.exit(1);
});
