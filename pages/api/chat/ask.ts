import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../src/utils/server/aviation_api_bridge";
import { requireApiAuth } from "../../../src/utils/server/require_api_auth";
import { enforceRateLimit, normalizeOptionalText } from "../../../src/utils/server/api_security";

type AskRequestBody = {
  session_id?: string;
  message?: string;
  strict_mode?: boolean;
  target_document?: string;
  model?: string;
  store?: boolean;
};

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  if (
    !enforceRateLimit(req, res, {
      namespace: "chat-ask",
      max: 30,
      windowMs: 5 * 60 * 1000,
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
    const body: AskRequestBody = req.body || {};
    const message = normalizeOptionalText(body.message, 4000);
    if (!message) {
      return res.status(400).json({
        success: false,
        error: "Field 'message' is required.",
      });
    }

    const args = ["ask", "--message", message];

    if (body.session_id) {
      args.push("--session-id", normalizeOptionalText(body.session_id, 128));
    }
    if (typeof body.strict_mode === "boolean") {
      args.push("--strict-mode", String(body.strict_mode));
    }
    if (body.target_document) {
      args.push("--target-document", normalizeOptionalText(body.target_document, 256));
    }
    if (body.model) {
      args.push("--model", normalizeOptionalText(body.model, 128));
    }
    if (typeof body.store === "boolean") {
      args.push("--store", String(body.store));
    }

    const result = await runAviationApiCommand(args);
    if (result.success !== true) {
      return res.status(500).json(result);
    }

    return res.status(200).json(result);
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
