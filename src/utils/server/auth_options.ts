import type { NextAuthOptions } from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import crypto from "crypto";

const defaultName = "AviationRAG User";
const fallbackSessionEmail = "user@aviationrag.local";

type AuthAttemptBucket = {
  count: number;
  resetAt: number;
};

const authAttemptBuckets = new Map<string, AuthAttemptBucket>();
const AUTH_ATTEMPT_WINDOW_MS = 15 * 60 * 1000;
const AUTH_ATTEMPT_MAX = 10;

function maybeCleanupAttemptBuckets(now: number) {
  authAttemptBuckets.forEach((bucket, key) => {
    if (bucket.resetAt <= now) {
      authAttemptBuckets.delete(key);
    }
  });
}

function getClientIpFromHeaders(rawHeaders: unknown): string {
  const headers = (rawHeaders || {}) as Record<string, string | string[] | undefined>;
  const forwarded = headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.trim()) {
    return forwarded.split(",")[0].trim();
  }
  if (Array.isArray(forwarded) && forwarded.length > 0) {
    return String(forwarded[0] || "").split(",")[0].trim();
  }
  const realIp = headers["x-real-ip"];
  if (typeof realIp === "string" && realIp.trim()) {
    return realIp.trim();
  }
  return "unknown";
}

function attemptKey(email: string, ip: string) {
  return `${email}:${ip}`;
}

function isBlocked(key: string, now: number): boolean {
  const bucket = authAttemptBuckets.get(key);
  return Boolean(bucket && bucket.resetAt > now && bucket.count >= AUTH_ATTEMPT_MAX);
}

function registerFailedAttempt(key: string, now: number) {
  const bucket = authAttemptBuckets.get(key);
  if (!bucket || bucket.resetAt <= now) {
    authAttemptBuckets.set(key, { count: 1, resetAt: now + AUTH_ATTEMPT_WINDOW_MS });
    return;
  }
  authAttemptBuckets.set(key, { ...bucket, count: bucket.count + 1 });
}

function clearAttempts(key: string) {
  authAttemptBuckets.delete(key);
}

function secureCompare(a: string, b: string): boolean {
  const left = Buffer.from(String(a || ""), "utf8");
  const right = Buffer.from(String(b || ""), "utf8");
  if (left.length !== right.length) {
    return false;
  }
  return crypto.timingSafeEqual(left, right);
}

function verifyPassword(password: string): boolean {
  const expectedHash = String(process.env.APP_AUTH_PASSWORD_HASH || "").trim();
  if (expectedHash.startsWith("sha256:")) {
    const expectedDigest = expectedHash.slice("sha256:".length).trim().toLowerCase();
    if (!expectedDigest) {
      return false;
    }
    const digest = crypto.createHash("sha256").update(password, "utf8").digest("hex");
    return secureCompare(digest, expectedDigest);
  }

  const expectedPassword = String(process.env.APP_AUTH_PASSWORD || "").trim();
  if (!expectedPassword) {
    return false;
  }
  return secureCompare(password, expectedPassword);
}

export const authOptions: NextAuthOptions = {
  session: { strategy: "jwt" },
  pages: { signIn: "/auth/signin" },
  providers: [
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials, req) {
        const expectedEmail = String(process.env.APP_AUTH_EMAIL || "").trim().toLowerCase();
        const now = Date.now();
        maybeCleanupAttemptBuckets(now);

        if (!expectedEmail) {
          return null;
        }

        const ip = getClientIpFromHeaders(req?.headers);
        const key = attemptKey(expectedEmail, ip);
        if (isBlocked(key, now)) {
          return null;
        }

        const email = String(credentials?.email || "").trim().toLowerCase();
        const password = String(credentials?.password || "").trim();

        if (!email || !password) {
          registerFailedAttempt(key, now);
          return null;
        }

        if (email !== expectedEmail || !verifyPassword(password)) {
          registerFailedAttempt(key, now);
          return null;
        }
        clearAttempts(key);

        return {
          id: "local-user",
          name: process.env.APP_AUTH_NAME || defaultName,
          email: expectedEmail,
        };
      },
    }),
  ],
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.name = user.name;
        token.email = user.email;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.name = String(token.name || session.user.name || defaultName);
        session.user.email = String(token.email || session.user.email || fallbackSessionEmail);
      }
      return session;
    },
  },
};
