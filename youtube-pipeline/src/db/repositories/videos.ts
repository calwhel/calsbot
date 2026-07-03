import type {
  CostSummaryRow,
  PendingVideoView,
  VideoRecord,
  VideoStatus,
} from "../../types/video";
import { query, queryOne } from "../pool";

export class VideoRepository {
  async createProcessing(
    channelId: string,
    topic: string | null,
  ): Promise<VideoRecord> {
    const row = await queryOne<VideoRecord>(
      `
      INSERT INTO videos (channel_id, topic, status)
      VALUES ($1, $2, 'processing')
      RETURNING *
      `,
      [channelId, topic],
    );

    if (!row) {
      throw new Error("Failed to create video record");
    }

    return row;
  }

  async markPrivate(
    videoId: string,
    data: {
      topic: string;
      title: string;
      youtubeVideoId: string;
      costUsd: number;
    },
  ): Promise<VideoRecord | null> {
    return queryOne<VideoRecord>(
      `
      UPDATE videos
      SET
        topic = $2,
        title = $3,
        status = 'private',
        youtube_video_id = $4,
        cost_usd = $5
      WHERE id = $1
      RETURNING *
      `,
      [
        videoId,
        data.topic,
        data.title,
        data.youtubeVideoId,
        data.costUsd,
      ],
    );
  }

  async markFailed(videoId: string): Promise<void> {
    await query(
      `UPDATE videos SET status = 'failed' WHERE id = $1`,
      [videoId],
    );
  }

  async markPublished(videoId: string): Promise<VideoRecord | null> {
    return queryOne<VideoRecord>(
      `
      UPDATE videos
      SET
        status = 'published',
        published_at = NOW()
      WHERE id = $1
      RETURNING *
      `,
      [videoId],
    );
  }

  async findById(videoId: string): Promise<VideoRecord | null> {
    return queryOne<VideoRecord>(
      `SELECT * FROM videos WHERE id = $1`,
      [videoId],
    );
  }

  async listPending(): Promise<PendingVideoView[]> {
    const rows = await query<{
      id: string;
      channel_id: string;
      channel_name: string;
      topic: string | null;
      title: string | null;
      status: VideoStatus;
      youtube_video_id: string | null;
      created_at: Date;
      cost_usd: string;
      view_count: string;
    }>(
      `
      SELECT
        v.id,
        v.channel_id,
        c.name AS channel_name,
        v.topic,
        v.title,
        v.status,
        v.youtube_video_id,
        v.created_at,
        v.cost_usd,
        v.view_count
      FROM videos v
      INNER JOIN channels c ON c.id = v.channel_id
      WHERE v.status = 'private'
      ORDER BY v.created_at DESC
      `,
    );

    return rows.map((row) => ({
      id: row.id,
      channel_id: row.channel_id,
      channel_name: row.channel_name,
      topic: row.topic,
      title: row.title,
      status: row.status,
      youtube_video_id: row.youtube_video_id,
      created_at: row.created_at.toISOString(),
      cost_usd: Number(row.cost_usd),
      view_count: Number(row.view_count),
    }));
  }

  async getCostSummary(): Promise<CostSummaryRow[]> {
    const rows = await query<{
      channel_id: string;
      channel_name: string;
      month: Date;
      total_cost_usd: string;
      video_count: string;
    }>(
      `
      SELECT
        v.channel_id,
        c.name AS channel_name,
        date_trunc('month', v.created_at) AS month,
        COALESCE(SUM(v.cost_usd), 0)::text AS total_cost_usd,
        COUNT(*)::text AS video_count
      FROM videos v
      INNER JOIN channels c ON c.id = v.channel_id
      GROUP BY v.channel_id, c.name, date_trunc('month', v.created_at)
      ORDER BY month DESC, channel_name ASC
      `,
    );

    return rows.map((row) => ({
      channel_id: row.channel_id,
      channel_name: row.channel_name,
      month: row.month.toISOString().slice(0, 7),
      total_cost_usd: Number(row.total_cost_usd),
      video_count: Number(row.video_count),
    }));
  }
}
