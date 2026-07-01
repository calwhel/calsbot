import express, { type NextFunction, type Request, type Response } from "express";

import { loadConfig } from "./config";
import { PipelineOrchestrator } from "./pipeline";
import { streamTransientAsset } from "./utils/assets";

const config = loadConfig();
const app = express();
const orchestrator = new PipelineOrchestrator(config);

let pipelineRunning = false;

app.use(express.json({ limit: "1mb" }));

function authenticate(
  req: Request,
  res: Response,
  next: NextFunction,
): void {
  const token = req.header("x-auth-token") ?? req.header("authorization");
  const normalized =
    token?.startsWith("Bearer ") === true ? token.slice(7) : token;

  if (!normalized || normalized !== config.authToken) {
    res.status(401).json({ error: "Unauthorized" });
    return;
  }

  next();
}

app.get("/health", (_req, res) => {
  res.status(200).json({
    status: "ok",
    pipelineRunning,
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

async function handleRunPipeline(
  req: Request,
  res: Response,
): Promise<void> {
  if (pipelineRunning) {
    res.status(409).json({ error: "Pipeline is already running" });
    return;
  }

  pipelineRunning = true;
  const topic =
    typeof req.body?.topic === "string" ? req.body.topic : undefined;

  try {
    const result = await orchestrator.run({ topic });
    res.status(200).json({
      success: true,
      result,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error("[pipeline] failed:", message);
    res.status(500).json({
      success: false,
      error: message,
    });
  } finally {
    pipelineRunning = false;
  }
}

app.post("/api/run-pipeline", authenticate, (req, res) => {
  void handleRunPipeline(req, res);
});

app.get("/api/run-pipeline", authenticate, (req, res) => {
  void handleRunPipeline(req, res);
});

app.listen(config.port, () => {
  console.log(
    `YouTube pipeline server listening on port ${config.port}`,
  );
});
