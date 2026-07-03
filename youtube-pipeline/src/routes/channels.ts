import type { Request, Response, Router } from "express";

import type { PlatformConfig } from "../config";
import { ChannelRepository } from "../db/repositories/channels";
import type {
  CreateChannelInput,
  UpdateChannelInput,
} from "../types/channel";

function parseCreateChannelBody(body: unknown): CreateChannelInput {
  if (!body || typeof body !== "object") {
    throw new Error("Request body must be a JSON object");
  }

  const record = body as Record<string, unknown>;
  const requiredStringFields = [
    "name",
    "niche_prompt",
    "youtube_client_id",
    "youtube_client_secret",
    "youtube_refresh_token",
    "elevenlabs_voice_id",
    "creatomate_template_id",
  ] as const;

  for (const field of requiredStringFields) {
    if (typeof record[field] !== "string" || record[field].trim() === "") {
      throw new Error(`Missing or invalid field: ${field}`);
    }
  }

  return {
    name: (record.name as string).trim(),
    niche_prompt: (record.niche_prompt as string).trim(),
    youtube_client_id: (record.youtube_client_id as string).trim(),
    youtube_client_secret: (record.youtube_client_secret as string).trim(),
    youtube_refresh_token: (record.youtube_refresh_token as string).trim(),
    elevenlabs_voice_id: (record.elevenlabs_voice_id as string).trim(),
    creatomate_template_id: (record.creatomate_template_id as string).trim(),
    upload_frequency:
      typeof record.upload_frequency === "string"
        ? record.upload_frequency.trim()
        : undefined,
    monthly_budget_usd:
      typeof record.monthly_budget_usd === "number"
        ? record.monthly_budget_usd
        : undefined,
    status:
      record.status === "active" || record.status === "paused"
        ? record.status
        : undefined,
  };
}

function parseUpdateChannelBody(body: unknown): UpdateChannelInput {
  if (!body || typeof body !== "object") {
    throw new Error("Request body must be a JSON object");
  }

  const record = body as Record<string, unknown>;
  const input: UpdateChannelInput = {};

  const stringFields = [
    "name",
    "niche_prompt",
    "youtube_client_id",
    "youtube_client_secret",
    "youtube_refresh_token",
    "elevenlabs_voice_id",
    "creatomate_template_id",
    "upload_frequency",
  ] as const;

  for (const field of stringFields) {
    if (record[field] !== undefined) {
      if (typeof record[field] !== "string" || record[field].trim() === "") {
        throw new Error(`Invalid field: ${field}`);
      }
      input[field] = record[field].trim();
    }
  }

  if (record.monthly_budget_usd !== undefined) {
    if (typeof record.monthly_budget_usd !== "number") {
      throw new Error("Invalid field: monthly_budget_usd");
    }
    input.monthly_budget_usd = record.monthly_budget_usd;
  }

  if (record.status !== undefined) {
    if (record.status !== "active" && record.status !== "paused") {
      throw new Error("Invalid field: status");
    }
    input.status = record.status;
  }

  return input;
}

export function createChannelRoutes(
  router: Router,
  platform: PlatformConfig,
): void {
  const channels = new ChannelRepository(platform.encryptionKey);

  router.post("/channels", async (req: Request, res: Response) => {
    try {
      const input = parseCreateChannelBody(req.body);
      const channel = await channels.create(input);
      res.status(201).json({ channel });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(400).json({ error: message });
    }
  });

  router.get("/channels", async (_req, res) => {
    const channelList = await channels.listAll();
    res.status(200).json({ channels: channelList });
  });

  router.get("/channels/:id", async (req, res) => {
    const channel = await channels.getPublicView(req.params.id);
    if (!channel) {
      res.status(404).json({ error: "Channel not found" });
      return;
    }

    res.status(200).json({ channel });
  });

  router.patch("/channels/:id", async (req, res) => {
    try {
      const input = parseUpdateChannelBody(req.body);
      const channel = await channels.update(req.params.id, input);
      if (!channel) {
        res.status(404).json({ error: "Channel not found" });
        return;
      }

      res.status(200).json({ channel });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      res.status(400).json({ error: message });
    }
  });

  router.delete("/channels/:id", async (req, res) => {
    const deleted = await channels.delete(req.params.id);
    if (!deleted) {
      res.status(404).json({ error: "Channel not found" });
      return;
    }

    res.status(200).json({ success: true });
  });
}
