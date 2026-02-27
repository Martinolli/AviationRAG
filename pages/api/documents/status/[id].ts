import type { NextApiRequest, NextApiResponse } from "next";
import { requireApiAuth } from "../../../../src/utils/server/require_api_auth";
import {
  enforceRateLimit,
  normalizeOptionalText,
} from "../../../../src/utils/server/api_security";
import { getUploadJob } from "../../../../src/utils/server/document_upload_jobs";

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (
    !enforceRateLimit(req, res, {
      namespace: "documents-status",
      max: 120,
      windowMs: 5 * 60 * 1000,
    })
  ) {
    return;
  }

  const session = await requireApiAuth(req, res);
  if (!session) {
    return;
  }

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  }

  const id = normalizeOptionalText(req.query.id, 128);
  if (!id) {
    return res.status(400).json({ success: false, error: "Upload id is required." });
  }

  const job = getUploadJob(id);
  if (!job) {
    return res.status(404).json({ success: false, error: "Upload job not found." });
  }

  return res.status(200).json({
    success: true,
    job,
  });
}
