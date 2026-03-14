import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import path from "path";

type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
interface JsonObject {
  [key: string]: JsonValue;
}
interface JsonArray extends Array<JsonValue> {}
type BridgeMode = "worker" | "http";

type PendingRequest = {
  resolve: (value: JsonObject) => void;
  reject: (error: Error) => void;
  timeout: NodeJS.Timeout;
};

let workerProcess: ChildProcessWithoutNullStreams | null = null;
let stdoutBuffer = "";
let stderrBuffer = "";
const pendingRequests = new Map<string, PendingRequest>();
let requestCounter = 0;

function nextRequestId(): string {
  requestCounter += 1;
  return `req_${Date.now()}_${requestCounter}`;
}

function parseBoolean(value: string): boolean {
  const normalized = String(value).trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes" || normalized === "on";
}

function currentBridgeMode(): BridgeMode {
  const raw = String(process.env.AVIATION_API_MODE || "worker").trim().toLowerCase();
  return raw === "http" ? "http" : "worker";
}

function parseArgsToPayload(args: string[]): JsonObject {
  const action = args[0];
  const payload: JsonObject = { action };
  const booleanKeys = new Set(["strict_mode", "store", "pinned", "purge_history"]);

  for (let i = 1; i < args.length; i += 1) {
    const keyToken = args[i];
    if (!keyToken.startsWith("--")) {
      continue;
    }
    const rawKey = keyToken.slice(2).replace(/-/g, "_");
    const rawValue = i + 1 < args.length ? args[i + 1] : "";

    let value: JsonValue = rawValue;
    if (booleanKeys.has(rawKey)) {
      value = parseBoolean(rawValue);
    } else if (rawKey === "limit") {
      const parsed = Number(rawValue);
      value = Number.isFinite(parsed) ? parsed : 10;
    }

    payload[rawKey] = value;
    i += 1;
  }

  return payload;
}

function rejectAllPending(error: Error) {
  pendingRequests.forEach((pending, id) => {
    clearTimeout(pending.timeout);
    pending.reject(new Error(`${error.message} [request_id=${id}]`));
  });
  pendingRequests.clear();
}

function ensureWorker(): ChildProcessWithoutNullStreams {
  if (workerProcess && !workerProcess.killed) {
    return workerProcess;
  }

  const pythonExecutable = process.env.PYTHON_EXECUTABLE || "python";
  const workerPath = path.join(
    process.cwd(),
    "src",
    "scripts",
    "py_files",
    "aviationai_worker.py",
  );

  workerProcess = spawn(pythonExecutable, [workerPath], {
    cwd: process.cwd(),
    stdio: ["pipe", "pipe", "pipe"],
  });

  stdoutBuffer = "";
  stderrBuffer = "";

  workerProcess.stdout.on("data", (chunk) => {
    stdoutBuffer += String(chunk);
    const lines = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      let parsed: JsonObject;
      try {
        parsed = JSON.parse(trimmed) as JsonObject;
      } catch {
        continue;
      }

      const requestId = parsed.id ? String(parsed.id) : "";
      if (!requestId) {
        continue;
      }

      const pending = pendingRequests.get(requestId);
      if (!pending) {
        continue;
      }

      clearTimeout(pending.timeout);
      pendingRequests.delete(requestId);
      pending.resolve(parsed);
    }
  });

  workerProcess.stderr.on("data", (chunk) => {
    stderrBuffer += String(chunk);
  });

  workerProcess.on("error", (error) => {
    const wrapped = new Error(`aviationai_worker.py process error: ${error.message}`);
    rejectAllPending(wrapped);
    workerProcess = null;
  });

  workerProcess.on("close", (code) => {
    const wrapped = new Error(
      `aviationai_worker.py exited with code ${code}. stderr: ${stderrBuffer || "n/a"}`,
    );
    rejectAllPending(wrapped);
    workerProcess = null;
  });

  return workerProcess;
}

async function runHttpBridgeCommand(payload: JsonObject, timeoutMs: number): Promise<JsonObject> {
  const baseUrl = String(process.env.AVIATION_API_HTTP_URL || "").trim();
  if (!baseUrl) {
    throw new Error("AVIATION_API_MODE is http but AVIATION_API_HTTP_URL is not set.");
  }

  const token = String(process.env.AVIATION_API_HTTP_TOKEN || "").trim();
  const endpoint = baseUrl.endsWith("/")
    ? `${baseUrl}command`
    : `${baseUrl}/command`;

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const text = await response.text();
    let parsed: JsonObject = {};
    if (text) {
      try {
        parsed = JSON.parse(text) as JsonObject;
      } catch {
        parsed = { success: false, error: `Invalid JSON from HTTP bridge: ${text}` };
      }
    }

    if (!response.ok) {
      throw new Error(
        `HTTP bridge request failed (${response.status}): ${String(parsed.error || response.statusText)}`,
      );
    }

    if (parsed.success === false) {
      throw new Error(String(parsed.error || "Aviation API HTTP bridge error"));
    }

    return parsed;
  } finally {
    clearTimeout(timeout);
  }
}

export async function runAviationApiCommand(args: string[]): Promise<JsonObject> {
  const payload = parseArgsToPayload(args);
  const requestId = nextRequestId();
  payload.id = requestId;

  const timeoutMs = Number(process.env.AVIATION_API_TIMEOUT_MS || 180000);
  if (currentBridgeMode() === "http") {
    return runHttpBridgeCommand(payload, timeoutMs);
  }

  const processRef = ensureWorker();
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      pendingRequests.delete(requestId);
      reject(new Error(`aviationai_worker.py request timeout after ${timeoutMs}ms`));
    }, timeoutMs);

    pendingRequests.set(requestId, {
      resolve: (value) => {
        if (value.success === false) {
          reject(new Error(String(value.error || "Aviation API error")));
          return;
        }
        resolve(value);
      },
      reject,
      timeout,
    });

    processRef.stdin.write(`${JSON.stringify(payload)}\n`);
  });
}
