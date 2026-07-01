CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS channels (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  niche_prompt TEXT NOT NULL,
  youtube_client_id TEXT NOT NULL,
  youtube_client_secret TEXT NOT NULL,
  youtube_refresh_token TEXT NOT NULL,
  elevenlabs_voice_id TEXT NOT NULL,
  creatomate_template_id TEXT NOT NULL,
  upload_frequency TEXT NOT NULL DEFAULT '0 14 * * *',
  monthly_budget_usd NUMERIC(10, 2) NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'paused' CHECK (status IN ('active', 'paused')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS videos (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  topic TEXT,
  title TEXT,
  status TEXT NOT NULL DEFAULT 'processing' CHECK (
    status IN ('processing', 'private', 'published', 'failed')
  ),
  youtube_video_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  cost_usd NUMERIC(10, 4) DEFAULT 0,
  view_count BIGINT DEFAULT 0,
  published_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS topics_used (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  channel_id UUID NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
  topic_hash TEXT NOT NULL,
  topic_text TEXT NOT NULL,
  used_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (channel_id, topic_hash)
);

CREATE TABLE IF NOT EXISTS channel_stats (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  channel_id UUID NOT NULL UNIQUE REFERENCES channels(id) ON DELETE CASCADE,
  subs_count BIGINT DEFAULT 0,
  watch_hours_total NUMERIC(12, 2) DEFAULT 0,
  monetization_eligible BOOLEAN DEFAULT FALSE,
  last_checked_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
CREATE INDEX IF NOT EXISTS idx_videos_channel_created ON videos(channel_id, created_at);
CREATE INDEX IF NOT EXISTS idx_topics_used_channel_id ON topics_used(channel_id);
