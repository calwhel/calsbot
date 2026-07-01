export interface VideoScene {
  voiceover_text: string;
  visual_prompt: string;
  overlay_text: string;
}

export interface VideoPayload {
  title: string;
  description: string;
  tags: string[];
  thumbnail_prompt: string;
  thumbnail_text: string;
  scenes: VideoScene[];
}

export interface PipelineResult {
  videoId: string;
  videoUrl: string;
  title: string;
  privacyStatus: string;
  voiceoverPath: string;
  renderedVideoUrl: string;
}
