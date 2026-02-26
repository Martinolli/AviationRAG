import type { NextApiRequest, NextApiResponse } from "next";

type Bucket = {
  count: number;
  resetAt: number;
};

type RateLimitOptions = {
  namespace: string;
  max: number;
  windowMs: number;
};

const rateBuckets = new Map<string, Bucket>();

let cleanupCounter = 0;

function maybeCleanupExpiredBuckets(now: number) {
  cleanupCounter += 1;
  if (cleanupCounter % 200 !== 0) {
    return;
  }

  rateBuckets.forEach((bucket, key) => {
    if (bucket.resetAt <= now) {
      rateBuckets.delete(key);
    }
  });
}

function clientIpFromRequest(req: NextApiRequest): string {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.trim()) {
    return forwarded.split(",")[0].trim();
  }
  if (Array.isArray(forwarded) && forwarded.length > 0) {
    return String(forwarded[0] || "").split(",")[0].trim();
  }
  return String(req.socket?.remoteAddress || "unknown");
}

export function enforceRateLimit(
  req: NextApiRequest,
  res: NextApiResponse,
  options: RateLimitOptions,
): boolean {
  if (process.env.DISABLE_API_RATE_LIMIT === "true") {
    return true;
  }

  const now = Date.now();
  maybeCleanupExpiredBuckets(now);

  const ip = clientIpFromRequest(req);
  const key = `${options.namespace}:${ip}`;
  const current = rateBuckets.get(key);

  if (!current || current.resetAt <= now) {
    rateBuckets.set(key, { count: 1, resetAt: now + options.windowMs });
    return true;
  }

  if (current.count >= options.max) {
    const retryAfterSeconds = Math.max(1, Math.ceil((current.resetAt - now) / 1000));
    res.setHeader("Retry-After", String(retryAfterSeconds));
    res.status(429).json({
      success: false,
      error: "Too many requests. Please retry shortly.",
    });
    return false;
  }

  current.count += 1;
  rateBuckets.set(key, current);
  return true;
}

export function readSingleQueryValue(rawValue: string | string[] | undefined): string {
  return String(Array.isArray(rawValue) ? rawValue[0] : rawValue || "");
}

export function normalizeOptionalText(value: unknown, maxLength = 256): string {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  return text.slice(0, maxLength);
}
