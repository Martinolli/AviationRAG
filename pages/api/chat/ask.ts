import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../src/utils/server/aviation_api_bridge";

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
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  }

  try {
    const body: AskRequestBody = req.body || {};
    const message = String(body.message || "").trim();
    if (!message) {
      return res.status(400).json({
        success: false,
        error: "Field 'message' is required.",
      });
    }

    const args = ["ask", "--message", message];

    if (body.session_id) {
      args.push("--session-id", String(body.session_id));
    }
    if (typeof body.strict_mode === "boolean") {
      args.push("--strict-mode", String(body.strict_mode));
    }
    if (body.target_document) {
      args.push("--target-document", String(body.target_document));
    }
    if (body.model) {
      args.push("--model", String(body.model));
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

