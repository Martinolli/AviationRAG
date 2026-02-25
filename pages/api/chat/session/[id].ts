import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../../src/utils/server/aviation_api_bridge";

type SessionUpdateBody = {
  title?: string;
  pinned?: boolean;
  purge_history?: boolean;
};

function parseBoolean(rawValue: string | string[] | undefined, fallback: boolean) {
  const value = String(Array.isArray(rawValue) ? rawValue[0] : rawValue || "").trim().toLowerCase();
  if (!value) return fallback;
  return value === "1" || value === "true" || value === "yes" || value === "on";
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const sessionId = String(req.query.id || "").trim();
  if (!sessionId) {
    return res.status(400).json({ success: false, error: "Path parameter 'id' is required." });
  }

  try {
    if (req.method === "PATCH") {
      const body: SessionUpdateBody = req.body || {};
      const title = String(body.title || "").trim();
      const args = ["session_upsert", "--session-id", sessionId];

      if (title) {
        args.push("--title", title);
      }
      if (typeof body.pinned === "boolean") {
        args.push("--pinned", String(body.pinned));
      }

      const result = await runAviationApiCommand(args);
      return res.status(200).json(result);
    }

    if (req.method === "DELETE") {
      const body: SessionUpdateBody = req.body || {};
      const purgeHistory =
        typeof body.purge_history === "boolean"
          ? body.purge_history
          : parseBoolean(req.query.purge_history, true);

      const result = await runAviationApiCommand([
        "session_delete",
        "--session-id",
        sessionId,
        "--purge-history",
        String(purgeHistory),
      ]);
      return res.status(200).json(result);
    }

    res.setHeader("Allow", "PATCH, DELETE");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}

