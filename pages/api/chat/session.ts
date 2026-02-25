import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../src/utils/server/aviation_api_bridge";
import { requireApiAuth } from "../../../src/utils/server/require_api_auth";

type SessionRequestBody = {
  session_id?: string;
  title?: string;
  pinned?: boolean;
};

function parseLimit(rawValue: string | string[] | undefined, fallback: number) {
  const value = Array.isArray(rawValue) ? rawValue[0] : rawValue;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(Math.floor(parsed), 1), 200);
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const session = await requireApiAuth(req, res);
  if (!session) {
    return;
  }

  try {
    if (req.method === "GET") {
      const search = String(req.query.search || "").trim();
      const filter = String(req.query.filter || "all").trim().toLowerCase();
      const limit = parseLimit(req.query.limit, 50);

      const args = [
        "sessions_list",
        "--search",
        search,
        "--filter",
        filter,
        "--limit",
        String(limit),
      ];
      const result = await runAviationApiCommand(args);
      return res.status(200).json(result);
    }

    if (req.method === "POST") {
      const body: SessionRequestBody = req.body || {};
      const title = String(body.title || "").trim();
      const args = ["session_upsert"];

      if (body.session_id) {
        args.push("--session-id", String(body.session_id));
      }
      if (title) {
        args.push("--title", title);
      }
      if (typeof body.pinned === "boolean") {
        args.push("--pinned", String(body.pinned));
      }

      const result = await runAviationApiCommand(args);
      return res.status(201).json(result);
    }

    res.setHeader("Allow", "GET, POST");
    return res.status(405).json({ success: false, error: "Method Not Allowed" });
  } catch (error) {
    return res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    });
  }
}
