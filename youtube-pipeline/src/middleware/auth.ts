import type { NextFunction, Request, Response } from "express";

import type { PlatformConfig } from "../config";

export function createAuthMiddleware(config: PlatformConfig) {
  return function authenticate(
    req: Request,
    res: Response,
    next: NextFunction,
  ): void {
    const token = req.header("x-auth-token") ?? req.header("authorization");
    const normalized =
      token?.startsWith("Bearer ") === true ? token.slice(7) : token;

    if (!normalized || normalized !== config.authToken) {
      res.status(401).json({ error: "Unauthorized" });
      return;
    }

    next();
  };
}
