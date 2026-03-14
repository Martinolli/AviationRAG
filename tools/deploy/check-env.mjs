#!/usr/bin/env node
import fs from "fs";
import path from "path";
import process from "process";
import dotenv from "dotenv";

const profileArgIndex = process.argv.indexOf("--profile");
const fileArgIndex = process.argv.indexOf("--file");

const profile =
  profileArgIndex >= 0 && process.argv[profileArgIndex + 1]
    ? String(process.argv[profileArgIndex + 1]).trim().toLowerCase()
    : "local-http";

const envFile =
  fileArgIndex >= 0 && process.argv[fileArgIndex + 1]
    ? String(process.argv[fileArgIndex + 1]).trim()
    : ".env";

const resolvedEnvFile = path.resolve(process.cwd(), envFile);
if (!fs.existsSync(resolvedEnvFile)) {
  console.error(`ERROR: Env file not found: ${resolvedEnvFile}`);
  process.exit(1);
}

dotenv.config({ path: resolvedEnvFile, override: true });

const profiles = {
  "local-http": {
    required: [
      "OPENAI_API_KEY",
      "ASTRA_DB_SECURE_BUNDLE_PATH",
      "ASTRA_DB_APPLICATION_TOKEN",
      "ASTRA_DB_KEYSPACE",
      "NEXTAUTH_URL",
      "NEXTAUTH_SECRET",
      "APP_AUTH_EMAIL",
      "AVIATION_API_MODE",
      "AVIATION_API_HTTP_URL",
      "AVIATION_API_HTTP_TOKEN",
    ],
    expectedMode: "http",
    requireHttpsNextAuth: false,
    requireHttpsBridgeUrl: false,
    allowLocalBridgeUrl: true,
  },
  "vercel-http": {
    required: [
      "OPENAI_API_KEY",
      "ASTRA_DB_SECURE_BUNDLE_PATH",
      "ASTRA_DB_APPLICATION_TOKEN",
      "ASTRA_DB_KEYSPACE",
      "NEXTAUTH_URL",
      "NEXTAUTH_SECRET",
      "APP_AUTH_EMAIL",
      "APP_AUTH_PASSWORD_HASH",
      "AVIATION_API_MODE",
      "AVIATION_API_HTTP_URL",
      "AVIATION_API_HTTP_TOKEN",
    ],
    expectedMode: "http",
    requireHttpsNextAuth: true,
    requireHttpsBridgeUrl: true,
    allowLocalBridgeUrl: false,
  },
};

if (!profiles[profile]) {
  console.error(
    `ERROR: Unknown profile '${profile}'. Use one of: ${Object.keys(profiles).join(", ")}`,
  );
  process.exit(1);
}

const config = profiles[profile];
const missing = [];
const warnings = [];

for (const key of config.required) {
  if (!String(process.env[key] || "").trim()) {
    missing.push(key);
  }
}

const mode = String(process.env.AVIATION_API_MODE || "").trim().toLowerCase();
if (mode !== config.expectedMode) {
  missing.push(`AVIATION_API_MODE=${config.expectedMode}`);
}

const nextAuthUrl = String(process.env.NEXTAUTH_URL || "").trim();
if (nextAuthUrl) {
  if (config.requireHttpsNextAuth && !nextAuthUrl.startsWith("https://")) {
    missing.push("NEXTAUTH_URL must start with https:// for vercel-http profile");
  }
}

const bridgeUrl = String(process.env.AVIATION_API_HTTP_URL || "").trim();
if (bridgeUrl) {
  if (config.requireHttpsBridgeUrl && !bridgeUrl.startsWith("https://")) {
    missing.push("AVIATION_API_HTTP_URL must start with https:// for vercel-http profile");
  }
  if (!config.allowLocalBridgeUrl) {
    const normalized = bridgeUrl.toLowerCase();
    if (
      normalized.includes("localhost") ||
      normalized.includes("127.0.0.1") ||
      normalized.includes("0.0.0.0")
    ) {
      missing.push("AVIATION_API_HTTP_URL cannot use localhost/loopback for vercel-http profile");
    }
  }
}

const nextAuthSecret = String(process.env.NEXTAUTH_SECRET || "");
if (nextAuthSecret && nextAuthSecret.length < 24) {
  warnings.push("NEXTAUTH_SECRET should be at least 24 characters.");
}

if (String(process.env.APP_AUTH_PASSWORD || "").trim()) {
  warnings.push("APP_AUTH_PASSWORD is set. Prefer APP_AUTH_PASSWORD_HASH for safer deployments.");
}

if (missing.length > 0) {
  console.error(`\n[deploy:check] FAILED for profile '${profile}'`);
  for (const item of missing) {
    console.error(`- Missing or invalid: ${item}`);
  }
  if (warnings.length > 0) {
    console.error("\nWarnings:");
    for (const warning of warnings) {
      console.error(`- ${warning}`);
    }
  }
  process.exit(1);
}

console.log(`\n[deploy:check] OK for profile '${profile}' using ${resolvedEnvFile}`);
if (warnings.length > 0) {
  console.log("\nWarnings:");
  for (const warning of warnings) {
    console.log(`- ${warning}`);
  }
}

