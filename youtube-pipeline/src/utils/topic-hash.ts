import { createHash } from "node:crypto";

export function hashTopic(topic: string): string {
  return createHash("sha256")
    .update(topic.trim().toLowerCase())
    .digest("hex");
}
