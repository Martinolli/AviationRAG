import fs from "fs";
import path from "path";
import { randomUUID } from "crypto";

export type UploadStatus =
  | "uploaded"
  | "processing"
  | "embedded"
  | "available"
  | "needs_review"
  | "failed";

export type UploadJob = {
  id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  status: UploadStatus;
  message: string;
  created_at: string;
  updated_at: string;
};

const jobs = new Map<string, UploadJob>();
const jobsFilePath = path.join(process.cwd(), "logs", "upload_jobs.json");

function nowIso() {
  return new Date().toISOString();
}

function ensureJobsDirectory() {
  fs.mkdirSync(path.dirname(jobsFilePath), { recursive: true });
}

function writeJobsToDisk() {
  ensureJobsDirectory();
  const payload = Array.from(jobs.values()).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  fs.writeFileSync(jobsFilePath, `${JSON.stringify(payload, null, 2)}\n`, "utf8");
}

function loadJobsFromDisk() {
  if (!fs.existsSync(jobsFilePath)) {
    return;
  }

  try {
    const raw = fs.readFileSync(jobsFilePath, "utf8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) {
      return;
    }
    parsed.forEach((item) => {
      if (!item || typeof item !== "object") {
        return;
      }
      const id = String(item.id || "").trim();
      if (!id) {
        return;
      }
      jobs.set(id, {
        id,
        filename: String(item.filename || ""),
        mime_type: String(item.mime_type || ""),
        size_bytes: Number(item.size_bytes) || 0,
        status: String(item.status || "failed") as UploadStatus,
        message: String(item.message || ""),
        created_at: String(item.created_at || nowIso()),
        updated_at: String(item.updated_at || nowIso()),
      });
    });
  } catch {
    // Ignore corrupt runtime file; new jobs will rewrite it.
  }
}

loadJobsFromDisk();

export function createUploadJob(input: {
  filename: string;
  mime_type: string;
  size_bytes: number;
  message?: string;
}): UploadJob {
  const timestamp = nowIso();
  const job: UploadJob = {
    id: randomUUID(),
    filename: input.filename,
    mime_type: input.mime_type,
    size_bytes: input.size_bytes,
    status: "uploaded",
    message: input.message || "File uploaded successfully.",
    created_at: timestamp,
    updated_at: timestamp,
  };

  jobs.set(job.id, job);
  writeJobsToDisk();
  return job;
}

export function updateUploadJob(
  jobId: string,
  patch: Partial<Pick<UploadJob, "status" | "message">>,
): UploadJob | null {
  const existing = jobs.get(jobId);
  if (!existing) {
    return null;
  }

  const next: UploadJob = {
    ...existing,
    ...patch,
    updated_at: nowIso(),
  };
  jobs.set(jobId, next);
  writeJobsToDisk();
  return next;
}

export function getUploadJob(jobId: string): UploadJob | null {
  return jobs.get(jobId) || null;
}
