export type ChannelStatus = "active" | "paused";

export interface ChannelRecord {
  id: string;
  name: string;
  niche_prompt: string;
  youtube_client_id: string;
  youtube_client_secret: string;
  youtube_refresh_token: string;
  elevenlabs_voice_id: string;
  creatomate_template_id: string;
  upload_frequency: string;
  monthly_budget_usd: string;
  status: ChannelStatus;
  created_at: Date;
}

export interface ChannelStatsRecord {
  id: string;
  channel_id: string;
  subs_count: string;
  watch_hours_total: string;
  monetization_eligible: boolean;
  last_checked_at: Date | null;
}

export interface CreateChannelInput {
  name: string;
  niche_prompt: string;
  youtube_client_id: string;
  youtube_client_secret: string;
  youtube_refresh_token: string;
  elevenlabs_voice_id: string;
  creatomate_template_id: string;
  upload_frequency?: string;
  monthly_budget_usd?: number;
  status?: ChannelStatus;
}

export interface UpdateChannelInput {
  name?: string;
  niche_prompt?: string;
  youtube_client_id?: string;
  youtube_client_secret?: string;
  youtube_refresh_token?: string;
  elevenlabs_voice_id?: string;
  creatomate_template_id?: string;
  upload_frequency?: string;
  monthly_budget_usd?: number;
  status?: ChannelStatus;
}

export interface ChannelPublicView {
  id: string;
  name: string;
  niche_prompt: string;
  youtube_client_id: string;
  elevenlabs_voice_id: string;
  creatomate_template_id: string;
  upload_frequency: string;
  monthly_budget_usd: number;
  status: ChannelStatus;
  created_at: string;
  stats: {
    subs_count: number;
    watch_hours_total: number;
    monetization_eligible: boolean;
    last_checked_at: string | null;
  } | null;
}
