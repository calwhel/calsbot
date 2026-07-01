import { hashTopic } from "../../utils/topic-hash";
import { query } from "../pool";

export class TopicsUsedRepository {
  async listTopicTexts(channelId: string, limit = 100): Promise<string[]> {
    const rows = await query<{ topic_text: string }>(
      `
      SELECT topic_text
      FROM topics_used
      WHERE channel_id = $1
      ORDER BY used_at DESC
      LIMIT $2
      `,
      [channelId, limit],
    );

    return rows.map((row) => row.topic_text);
  }

  async recordTopic(channelId: string, topic: string): Promise<void> {
    await query(
      `
      INSERT INTO topics_used (channel_id, topic_hash, topic_text)
      VALUES ($1, $2, $3)
      ON CONFLICT (channel_id, topic_hash) DO UPDATE
      SET used_at = NOW(), topic_text = EXCLUDED.topic_text
      `,
      [channelId, hashTopic(topic), topic.trim()],
    );
  }
}
