import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../../src/utils/server/aviation_api_bridge";
import { requireApiAuth } from "../../../../src/utils/server/require_api_auth";

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  const session = await requireApiAuth(req, res);
  if (!session) {
    return;
  }

  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  }

  const sessionId = String(req.query.session_id || "").trim();
  if (!sessionId) {
    return res
      .status(400)
      .json({ success: false, error: "Path parameter 'session_id' is required." });
  }

  const limitRaw = Array.isArray(req.query.limit)
    ? req.query.limit[0]
    : req.query.limit;
  const limitValue = Number(limitRaw);
  const limit = Number.isFinite(limitValue)
    ? Math.min(Math.max(Math.floor(limitValue), 1), 50)
    : 10;

  try {
    const result = await runAviationApiCommand([
      "history",
      "--session-id",
      sessionId,
      "--limit",
      String(limit),
    ]);

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
