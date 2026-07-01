import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { writeFile } from "node:fs/promises";
import path from "node:path";

import dotenv from "dotenv";
import { google } from "googleapis";

const PROJECT_ROOT = path.resolve(__dirname, "..");
const REDIRECT_URI = "http://localhost:53682/oauth2callback";
const PORT = 53682;
const SCOPES = [
  "https://www.googleapis.com/auth/youtube.upload",
  "https://www.googleapis.com/auth/youtube.readonly",
];

dotenv.config({ path: path.join(PROJECT_ROOT, ".env") });

function requireEnv(name: string): string {
  const value = process.env[name]?.trim();
  if (!value) {
    console.error(
      `Error: ${name} is not set. Add it to ${path.join(PROJECT_ROOT, ".env")} and try again.`,
    );
    process.exit(1);
  }
  return value;
}

function parseCallbackUrl(requestUrl: string): URL {
  return new URL(requestUrl, REDIRECT_URI);
}

function sendHtml(res: ServerResponse, statusCode: number, body: string): void {
  res.writeHead(statusCode, { "Content-Type": "text/html; charset=utf-8" });
  res.end(body);
}

async function main(): Promise<void> {
  const clientId = requireEnv("YOUTUBE_CLIENT_ID");
  const clientSecret = requireEnv("YOUTUBE_CLIENT_SECRET");

  const oauth2Client = new google.auth.OAuth2(
    clientId,
    clientSecret,
    REDIRECT_URI,
  );

  const consentUrl = oauth2Client.generateAuthUrl({
    access_type: "offline",
    prompt: "consent",
    scope: SCOPES,
    redirect_uri: REDIRECT_URI,
  });

  console.log("Opening consent URL...");
  console.log("");
  console.log(consentUrl);
  console.log("");
  console.log(
    "Open the URL above in your browser and authorize with the Google account that owns the target YouTube channel.",
  );
  console.log("Waiting for authorization...");

  await new Promise<void>((resolve, reject) => {
    const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
      if (!req.url?.startsWith("/oauth2callback")) {
        sendHtml(res, 404, "<h1>Not found</h1>");
        return;
      }

      const callbackUrl = parseCallbackUrl(req.url);
      const error = callbackUrl.searchParams.get("error");
      const code = callbackUrl.searchParams.get("code");

      if (error) {
        sendHtml(
          res,
          400,
          `<h1>Authorization failed</h1><p>${error}</p><p>You can close this tab.</p>`,
        );
        server.close();
        reject(new Error(`OAuth authorization denied: ${error}`));
        return;
      }

      if (!code) {
        sendHtml(
          res,
          400,
          "<h1>Authorization failed</h1><p>No authorization code was returned.</p>",
        );
        server.close();
        reject(new Error("OAuth callback did not include a code query parameter."));
        return;
      }

      try {
        const { tokens } = await oauth2Client.getToken(code);
        const refreshToken = tokens.refresh_token;

        if (!refreshToken) {
          sendHtml(
            res,
            400,
            "<h1>Token exchange failed</h1><p>No refresh token was returned. Try revoking app access in your Google account and run the script again.</p>",
          );
          server.close();
          reject(
            new Error(
              "Token exchange succeeded but no refresh_token was returned. Re-run with prompt=consent after revoking prior app access.",
            ),
          );
          return;
        }

        const tokenFilePath = path.join(PROJECT_ROOT, ".refresh-token.txt");
        await writeFile(tokenFilePath, `${refreshToken}\n`, {
          encoding: "utf8",
          mode: 0o600,
        });

        console.log("Refresh token obtained. Add this to Railway as YOUTUBE_REFRESH_TOKEN:");
        console.log("");
        console.log(refreshToken);
        console.log("");
        console.log(`Saved refresh token to ${tokenFilePath}`);

        sendHtml(
          res,
          200,
          "<h1>Authorization complete</h1><p>Refresh token captured. You can close this tab and return to your terminal.</p>",
        );

        server.close(() => resolve());
      } catch (tokenError) {
        const message =
          tokenError instanceof Error ? tokenError.message : String(tokenError);
        sendHtml(
          res,
          500,
          `<h1>Token exchange failed</h1><p>${message}</p>`,
        );
        server.close();
        reject(
          tokenError instanceof Error
            ? tokenError
            : new Error(`Token exchange failed: ${message}`),
        );
      }
    });

    server.on("error", (serverError) => {
      reject(serverError);
    });

    server.listen(PORT, "127.0.0.1", () => {
      console.log(`Listening for OAuth callback on ${REDIRECT_URI}`);
    });
  });
}

main().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  console.error(`Error: ${message}`);
  process.exit(1);
});
