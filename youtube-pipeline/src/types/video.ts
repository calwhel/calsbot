export interface VideoScene {
  voiceover_text: string;
  visual_prompt: string;
  overlay_text: string;
}

export interface VideoPayload {
  topic: string;
  title: string;
  description: string;
  tags: string[];
  thumbnail_prompt: string;
  thumbnail_text: string;
  scenes: VideoScene[];
}

export type VideoStatus = "processing" | "private" | "published" | "failed";

export interface VideoRecord {
  id: string;
  channel_id: string;
  topic: string | null;
  title: string | null;
  status: VideoStatus;
  youtube_video_id: string | null;
  created_at: Date;
  cost_usd: string;
  view_count: string;
  published_at: Date | null;
}

export interface PendingVideoView {
  id: string;
  channel_id: string;
  channel_name: string;
  topic: string | null;
  title: string | null;
  status: VideoStatus;
  youtube_video_id: string | null;
  created_at: string;
  cost_usd: number;
  view_count: number;
}

export interface PipelineResult {
  videoId: string;
  videoUrl: string;
  title: string;
  topic: string;
  privacyStatus: string;
  dbVideoId: string;
  costUsd: number;
}

export interface CostSummaryRow {
  channel_id: string;
  channel_name: string;
  month: string;
  total_cost_usd: number;
  video_count: number;
}
