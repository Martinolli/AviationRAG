#!/usr/bin/env node

function parseArg(name) {
  const idx = process.argv.indexOf(name);
  if (idx >= 0 && process.argv[idx + 1]) {
    return String(process.argv[idx + 1]).trim();
  }
  return "";
}

function normalizeBaseUrl(url) {
  const value = String(url || "").trim();
  if (!value) return "";
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

function withTimeout(ms) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), ms);
  return { controller, timeout };
}

async function getJson(url, init = {}, timeoutMs = 10000) {
  const { controller, timeout } = withTimeout(timeoutMs);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    const text = await response.text();
    let payload = {};
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        throw new Error(`Invalid JSON from ${url}: ${text.slice(0, 200)}`);
      }
    }
    return { ok: response.ok, status: response.status, payload };
  } finally {
    clearTimeout(timeout);
  }
}

async function main() {
  const baseUrl = normalizeBaseUrl(parseArg("--url") || process.env.AVIATION_API_HTTP_URL || "");
  const token = String(parseArg("--token") || process.env.AVIATION_API_HTTP_TOKEN || "").trim();

  if (!baseUrl) {
    throw new Error("Missing bridge base URL. Use --url or AVIATION_API_HTTP_URL.");
  }
  if (!/^https?:\/\//i.test(baseUrl)) {
    throw new Error(`Bridge URL must start with http:// or https:// (received: ${baseUrl})`);
  }

  const healthUrl = `${baseUrl}/health`;
  const commandUrl = `${baseUrl}/command`;
  const headers = {
    "Content-Type": "application/json",
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };

  const health = await getJson(healthUrl, { method: "GET" }, 12000);
  if (!health.ok || health.payload?.success !== true) {
    throw new Error(
      `Bridge /health failed (status=${health.status}) payload=${JSON.stringify(health.payload)}`,
    );
  }

  const ping = await getJson(
    commandUrl,
    {
      method: "POST",
      headers,
      body: JSON.stringify({ action: "ping" }),
    },
    12000,
  );

  if (!ping.ok || ping.payload?.success !== true) {
    throw new Error(
      `Bridge /command ping failed (status=${ping.status}) payload=${JSON.stringify(ping.payload)}`,
    );
  }

  process.stdout.write(
    JSON.stringify(
      {
        success: true,
        bridge_url: baseUrl,
        health_status: health.status,
        ping_status: ping.status,
      },
      null,
      2,
    ) + "\n",
  );
}

main().catch((error) => {
  process.stderr.write(
    JSON.stringify(
      {
        success: false,
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    ) + "\n",
  );
  process.exit(1);
});

