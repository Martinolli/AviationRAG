import type { NextApiRequest, NextApiResponse } from "next";
import path from "path";
import fs from "fs";

type HealthResponse = {
  success: boolean;
  service: string;
  timestamp: string;
  bridge_mode: string;
  checks: {
    openai_api_key: boolean;
    astra_token: boolean;
    astra_bundle_path_set: boolean;
    astra_keyspace: boolean;
    python_bridge_script_exists: boolean;
    aviation_http_url_set: boolean;
    aviation_http_ping: boolean;
  };
  deep_check_requested: boolean;
  deep_check_error?: string;
};

function parseBooleanQuery(value: string | string[] | undefined): boolean {
  const raw = Array.isArray(value) ? value[0] : value;
  if (!raw) return false;
  const normalized = String(raw).trim().toLowerCase();
  return normalized === "1" || normalized === "true" || normalized === "yes" || normalized === "on";
}

async function runHttpBridgePingCheck(timeoutMs: number): Promise<{ ok: boolean; error?: string }> {
  const baseUrl = String(process.env.AVIATION_API_HTTP_URL || "").trim();
  if (!baseUrl) {
    return { ok: false, error: "AVIATION_API_HTTP_URL is empty." };
  }

  const endpoint = baseUrl.endsWith("/") ? `${baseUrl}command` : `${baseUrl}/command`;
  const token = String(process.env.AVIATION_API_HTTP_TOKEN || "").trim();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        id: `health_${Date.now()}`,
        action: "ping",
      }),
      signal: controller.signal,
    });

    const rawBody = await response.text();
    let payload: Record<string, unknown> = {};
    if (rawBody) {
      try {
        payload = JSON.parse(rawBody) as Record<string, unknown>;
      } catch {
        return {
          ok: false,
          error: `Bridge returned non-JSON response (${response.status}).`,
        };
      }
    }

    if (!response.ok) {
      return {
        ok: false,
        error: `Bridge HTTP ${response.status}: ${String(payload.error || response.statusText)}`,
      };
    }

    if (payload.success !== true) {
      return {
        ok: false,
        error: String(payload.error || "Bridge ping failed."),
      };
    }

    return { ok: true };
  } catch (error) {
    if (error instanceof Error) {
      return { ok: false, error: error.message };
    }
    return { ok: false, error: "Unknown bridge ping error." };
  } finally {
    clearTimeout(timeout);
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<HealthResponse>,
) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({
      success: false,
      service: "aviationrag-api",
      timestamp: new Date().toISOString(),
      bridge_mode: "worker",
      checks: {
        openai_api_key: false,
        astra_token: false,
        astra_bundle_path_set: false,
        astra_keyspace: false,
        python_bridge_script_exists: false,
        aviation_http_url_set: false,
        aviation_http_ping: false,
      },
      deep_check_requested: false,
    });
  }

  const scriptPath = path.join(
    process.cwd(),
    "src",
    "scripts",
    "py_files",
    "aviationai_api.py",
  );
  const bridgeMode = String(process.env.AVIATION_API_MODE || "worker").trim().toLowerCase();
  const isHttpMode = bridgeMode === "http";
  const deepCheckRequested = parseBooleanQuery(req.query.deep);

  let aviationHttpPing = !isHttpMode;
  let deepCheckError: string | undefined;

  if (isHttpMode && deepCheckRequested) {
    const pingResult = await runHttpBridgePingCheck(6000);
    aviationHttpPing = pingResult.ok;
    deepCheckError = pingResult.error;
  }

  return res.status(200).json({
    success: true,
    service: "aviationrag-api",
    timestamp: new Date().toISOString(),
    bridge_mode: isHttpMode ? "http" : "worker",
    checks: {
      openai_api_key: Boolean(process.env.OPENAI_API_KEY),
      astra_token: Boolean(process.env.ASTRA_DB_APPLICATION_TOKEN),
      astra_bundle_path_set: Boolean(process.env.ASTRA_DB_SECURE_BUNDLE_PATH),
      astra_keyspace: Boolean(process.env.ASTRA_DB_KEYSPACE),
      python_bridge_script_exists: isHttpMode ? true : fs.existsSync(scriptPath),
      aviation_http_url_set: isHttpMode ? Boolean(process.env.AVIATION_API_HTTP_URL) : true,
      aviation_http_ping: aviationHttpPing,
    },
    deep_check_requested: deepCheckRequested,
    ...(deepCheckError ? { deep_check_error: deepCheckError } : {}),
  });
}
