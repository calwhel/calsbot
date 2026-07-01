import Anthropic from "@anthropic-ai/sdk";

import type { AppConfig } from "../config";
import type { VideoPayload } from "../types/video";
import { withRetry } from "../utils/tmp";

const SYSTEM_PROMPT = `You are a YouTube video script generator. Return ONLY a single minified valid JSON object with no markdown, no code fences, and no commentary.

The JSON must match this TypeScript interface exactly:
interface VideoPayload {
  title: string;
  description: string;
  tags: string[];
  thumbnail_prompt: string;
  thumbnail_text: string;
  scenes: Array<{
    voiceover_text: string;
    visual_prompt: string;
    overlay_text: string;
  }>;
}

Rules:
- title: compelling, under 100 characters
- description: 2-4 paragraphs with SEO-friendly copy and a call to action
- tags: 8-15 relevant tags
- thumbnail_prompt: vivid image prompt for thumbnail generation
- thumbnail_text: short bold text for thumbnail overlay (max 6 words)
- scenes: 5-8 scenes; each voiceover_text 2-4 sentences; visual_prompt describes the scene imagery; overlay_text is a short on-screen caption
- Output must be parseable JSON only`;

function extractJsonObject(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenceMatch?.[1]) {
    return fenceMatch[1].trim();
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace >= 0 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }

  throw new Error("LLM response did not contain a JSON object");
}

function assertVideoPayload(value: unknown): VideoPayload {
  if (!value || typeof value !== "object") {
    throw new Error("LLM payload is not an object");
  }

  const payload = value as Record<string, unknown>;

  const requiredStrings = [
    "title",
    "description",
    "thumbnail_prompt",
    "thumbnail_text",
  ] as const;

  for (const key of requiredStrings) {
    if (typeof payload[key] !== "string" || payload[key].trim() === "") {
      throw new Error(`LLM payload missing or invalid field: ${key}`);
    }
  }

  if (!Array.isArray(payload.tags) || payload.tags.length === 0) {
    throw new Error("LLM payload missing or invalid field: tags");
  }

  if (!Array.isArray(payload.scenes) || payload.scenes.length === 0) {
    throw new Error("LLM payload missing or invalid field: scenes");
  }

  for (const [index, scene] of payload.scenes.entries()) {
    if (!scene || typeof scene !== "object") {
      throw new Error(`Scene ${index} is not an object`);
    }
    const sceneRecord = scene as Record<string, unknown>;
    for (const key of ["voiceover_text", "visual_prompt", "overlay_text"] as const) {
      if (
        typeof sceneRecord[key] !== "string" ||
        sceneRecord[key].trim() === ""
      ) {
        throw new Error(`Scene ${index} missing or invalid field: ${key}`);
      }
    }
  }

  return payload as unknown as VideoPayload;
}

export class LlmService {
  private readonly client: Anthropic;
  private readonly config: AppConfig;

  constructor(config: AppConfig) {
    this.config = config;
    this.client = new Anthropic({ apiKey: config.anthropic.apiKey });
  }

  async generateScript(topic?: string): Promise<VideoPayload> {
    const resolvedTopic = topic?.trim() || this.config.defaultTopic;

    return withRetry(
      async () => {
        const response = await this.client.messages.create({
          model: this.config.anthropic.model,
          max_tokens: 4096,
          temperature: 0.7,
          system: SYSTEM_PROMPT,
          messages: [
            {
              role: "user",
              content: `Generate a complete VideoPayload JSON for this topic: ${resolvedTopic}`,
            },
          ],
        });

        const textBlock = response.content.find(
          (block) => block.type === "text",
        );
        if (!textBlock || textBlock.type !== "text") {
          throw new Error("LLM response did not include text content");
        }

        const jsonString = extractJsonObject(textBlock.text);
        const parsed = JSON.parse(jsonString) as unknown;
        return assertVideoPayload(parsed);
      },
      {
        ...this.config.retry,
        label: "anthropic-script-generation",
      },
    );
  }
}
