import { readFile } from "node:fs/promises";
import path from "node:path";

import { Pool, type PoolClient, type QueryResultRow } from "pg";

let pool: Pool | null = null;

export function getDatabaseUrl(): string {
  const url =
    process.env.NEON_DATABASE_URL?.trim() ||
    process.env.DATABASE_URL?.trim() ||
    process.env.RAILWAY_DATABASE_URL?.trim();

  if (!url) {
    throw new Error(
      "Missing database URL. Set NEON_DATABASE_URL or DATABASE_URL.",
    );
  }

  return url;
}

export function getPool(): Pool {
  if (!pool) {
    pool = new Pool({
      connectionString: getDatabaseUrl(),
      ssl:
        process.env.PGSSLMODE === "disable"
          ? false
          : { rejectUnauthorized: false },
      max: 10,
    });
  }

  return pool;
}

export async function query<T extends QueryResultRow>(
  sql: string,
  params: unknown[] = [],
): Promise<T[]> {
  const result = await getPool().query<T>(sql, params);
  return result.rows;
}

export async function queryOne<T extends QueryResultRow>(
  sql: string,
  params: unknown[] = [],
): Promise<T | null> {
  const rows = await query<T>(sql, params);
  return rows[0] ?? null;
}

export async function withTransaction<T>(
  fn: (client: PoolClient) => Promise<T>,
): Promise<T> {
  const client = await getPool().connect();

  try {
    await client.query("BEGIN");
    const result = await fn(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK");
    throw error;
  } finally {
    client.release();
  }
}

export async function bootstrapSchema(): Promise<void> {
  const schemaPath = path.join(__dirname, "schema.sql");
  const schemaSql = await readFile(schemaPath, "utf8");
  await getPool().query(schemaSql);
  console.log("[db] schema bootstrap complete");
}

export async function closePool(): Promise<void> {
  if (pool) {
    await pool.end();
    pool = null;
  }
}
