import Anthropic from "@anthropic-ai/sdk";

import type { ServiceConfig } from "../config/channel-config";
import type { VideoPayload } from "../types/video";
import { withRetry } from "../utils/tmp";

function buildSystemPrompt(
  nichePrompt: string,
  excludedTopics: string[],
): string {
  const exclusionBlock =
    excludedTopics.length > 0
      ? `\nDo NOT reuse these previously covered topics:\n${excludedTopics
          .map((topic) => `- ${topic}`)
          .join("\n")}`
      : "";

  return `You are a YouTube video script generator for a channel with this niche and style:

${nichePrompt}

Return ONLY a single minified valid JSON object with no markdown, no code fences, and no commentary.

The JSON must match this TypeScript interface exactly:
interface VideoPayload {
  topic: string;
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
- topic: a specific, fresh episode topic within the channel niche (one concise sentence)
- title: compelling, under 100 characters
- description: 2-4 paragraphs with SEO-friendly copy and a call to action
- tags: 8-15 relevant tags
- thumbnail_prompt: vivid image prompt for thumbnail generation
- thumbnail_text: short bold text for thumbnail overlay (max 6 words)
- scenes: 5-8 scenes; each voiceover_text 2-4 sentences; visual_prompt describes the scene imagery; overlay_text is a short on-screen caption
- Stay strictly within the channel niche and tone
- Output must be parseable JSON only${exclusionBlock}`;
}

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
    "topic",
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

export interface GenerateScriptOptions {
  topic?: string;
  excludedTopics?: string[];
}

export class LlmService {
  private readonly client: Anthropic;
  private readonly config: ServiceConfig;

  constructor(config: ServiceConfig) {
    this.config = config;
    this.client = new Anthropic({ apiKey: config.anthropic.apiKey });
  }

  async generateScript(
    options: GenerateScriptOptions = {},
  ): Promise<VideoPayload> {
    const excludedTopics = options.excludedTopics ?? [];
    const systemPrompt = buildSystemPrompt(
      this.config.nichePrompt,
      excludedTopics,
    );
    const userPrompt = options.topic?.trim()
      ? `Generate a complete VideoPayload JSON for this specific topic: ${options.topic.trim()}`
      : "Generate a complete VideoPayload JSON choosing a fresh topic within the channel niche.";

    return withRetry(
      async () => {
        const response = await this.client.messages.create({
          model: this.config.anthropic.model,
          max_tokens: 4096,
          temperature: 0.7,
          system: systemPrompt,
          messages: [
            {
              role: "user",
              content: userPrompt,
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
        const payload = assertVideoPayload(parsed);

        if (options.topic?.trim()) {
          payload.topic = options.topic.trim();
        }

        return payload;
      },
      {
        ...this.config.retry,
        label: "anthropic-script-generation",
      },
    );
  }
}
