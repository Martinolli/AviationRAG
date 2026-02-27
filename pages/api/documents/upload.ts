import type { NextApiRequest, NextApiResponse } from "next";
import fs from "fs";
import path from "path";
import { spawn } from "child_process";
import formidable, { type File as FormidableFile, type Files } from "formidable";
import { requireApiAuth } from "../../../src/utils/server/require_api_auth";
import { enforceRateLimit } from "../../../src/utils/server/api_security";
import {
  createUploadJob,
  updateUploadJob,
  type UploadJob,
} from "../../../src/utils/server/document_upload_jobs";

const allowedExtensions = new Set([".pdf", ".docx"]);
const ingestSteps = [
  "Read Documents",
  "Chunk Documents",
  "Generate New Embeddings",
  "Store New Embeddings in AstraDB",
];

let ingestionQueue: Promise<void> = Promise.resolve();

function maxUploadBytes() {
  const parsed = Number(process.env.DOCUMENT_UPLOAD_MAX_MB || 25);
  const mb = Number.isFinite(parsed) && parsed > 0 ? parsed : 25;
  return Math.floor(mb * 1024 * 1024);
}

function sanitizeFilename(originalName: string, extension: string) {
  const withoutExt = path
    .basename(originalName, path.extname(originalName))
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  const baseName = withoutExt || `upload_${Date.now()}`;
  return `${baseName}${extension}`;
}

function uniqueDestinationPath(targetDir: string, fileName: string) {
  const extension = path.extname(fileName);
  const base = path.basename(fileName, extension);
  let candidate = fileName;
  let counter = 1;

  while (fs.existsSync(path.join(targetDir, candidate))) {
    candidate = `${base}_${Date.now()}_${counter}${extension}`;
    counter += 1;
  }
  return path.join(targetDir, candidate);
}

async function moveUploadedFile(tempPath: string, destinationPath: string) {
  try {
    await fs.promises.rename(tempPath, destinationPath);
    return;
  } catch (error: any) {
    if (String(error?.code || "") !== "EXDEV") {
      throw error;
    }
  }

  await fs.promises.copyFile(tempPath, destinationPath);
  await fs.promises.unlink(tempPath);
}

function parseMultipartForm(req: NextApiRequest): Promise<Files> {
  const form = formidable({
    multiples: false,
    maxFiles: 1,
    maxFileSize: maxUploadBytes(),
    keepExtensions: true,
  });

  return new Promise((resolve, reject) => {
    form.parse(req, (error, _fields, files) => {
      if (error) {
        reject(error);
        return;
      }
      resolve(files);
    });
  });
}

function pickFirstUploadedFile(files: Files): FormidableFile | null {
  for (const entry of Object.values(files)) {
    const file = Array.isArray(entry) ? entry[0] : entry;
    if (file?.filepath) {
      return file;
    }
  }
  return null;
}

function runManagerStep(stepName: string): Promise<void> {
  const pythonExecutable = process.env.PYTHON_EXECUTABLE || "python";
  const managerPath = path.join(
    process.cwd(),
    "src",
    "scripts",
    "py_files",
    "aviationrag_manager.py",
  );
  const timeoutMs = Number(process.env.DOCUMENT_UPLOAD_STEP_TIMEOUT_MS || 20 * 60 * 1000);

  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, [managerPath, "--step", stepName], {
      cwd: process.cwd(),
      env: process.env,
    });

    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Ingestion step timed out: ${stepName}`));
    }, timeoutMs);

    child.stderr.on("data", (chunk) => {
      stderr += String(chunk);
      if (stderr.length > 4000) {
        stderr = stderr.slice(-4000);
      }
    });

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });

    child.on("close", (code) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve();
        return;
      }
      reject(
        new Error(
          `Ingestion step failed: ${stepName} (exit ${code}). ${stderr || "No stderr output."}`,
        ),
      );
    });
  });
}

async function runIngestionPipeline(job: UploadJob) {
  if (String(process.env.DOCUMENT_UPLOAD_AUTO_INGEST || "true").toLowerCase() === "false") {
    updateUploadJob(job.id, {
      status: "uploaded",
      message: "Upload complete. Auto-ingestion disabled by DOCUMENT_UPLOAD_AUTO_INGEST=false.",
    });
    return;
  }

  try {
    for (const step of ingestSteps) {
      updateUploadJob(job.id, {
        status: "processing",
        message: `Running ingestion step: ${step}`,
      });
      // eslint-disable-next-line no-await-in-loop
      await runManagerStep(step);
    }

    updateUploadJob(job.id, {
      status: "embedded",
      message: "Embeddings stored successfully.",
    });
    updateUploadJob(job.id, {
      status: "available",
      message: "Document is available for chat retrieval.",
    });
  } catch (error) {
    updateUploadJob(job.id, {
      status: "failed",
      message: error instanceof Error ? error.message : "Ingestion failed.",
    });
  }
}

function enqueueIngestion(job: UploadJob) {
  ingestionQueue = ingestionQueue
    .then(() => runIngestionPipeline(job))
    .catch(() => {
      // Keep queue alive even if a prior task failed.
    });
}

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (
    !enforceRateLimit(req, res, {
      namespace: "documents-upload",
      max: 20,
      windowMs: 10 * 60 * 1000,
    })
  ) {
    return;
  }

  const session = await requireApiAuth(req, res);
  if (!session) {
    return;
  }

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  }

  try {
    const files = await parseMultipartForm(req);
    const uploaded = pickFirstUploadedFile(files);
    if (!uploaded) {
      return res.status(400).json({
        success: false,
        error: "No file received. Use multipart/form-data with field 'file'.",
      });
    }

    const originalName = String(uploaded.originalFilename || "").trim();
    const extension = path.extname(originalName).toLowerCase();
    if (!allowedExtensions.has(extension)) {
      return res.status(400).json({
        success: false,
        error: "Unsupported file type. Only .pdf and .docx are allowed.",
      });
    }

    const destinationDir = path.join(process.cwd(), "data", "documents");
    await fs.promises.mkdir(destinationDir, { recursive: true });

    const safeName = sanitizeFilename(originalName, extension);
    const destinationPath = uniqueDestinationPath(destinationDir, safeName);
    await moveUploadedFile(uploaded.filepath, destinationPath);

    const savedName = path.basename(destinationPath);
    const job = createUploadJob({
      filename: savedName,
      mime_type: String(uploaded.mimetype || "application/octet-stream"),
      size_bytes: Number(uploaded.size || 0),
      message: "File uploaded. Waiting for ingestion queue.",
    });
    enqueueIngestion(job);

    return res.status(201).json({
      success: true,
      upload_id: job.id,
      job,
    });
  } catch (error: any) {
    const tooLarge = String(error?.code || "").includes("1009");
    return res.status(tooLarge ? 413 : 500).json({
      success: false,
      error: tooLarge
        ? "Upload too large. Increase DOCUMENT_UPLOAD_MAX_MB or use a smaller file."
        : error instanceof Error
          ? error.message
          : "Upload failed.",
    });
  }
}
