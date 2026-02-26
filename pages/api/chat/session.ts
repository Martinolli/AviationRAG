import type { NextApiRequest, NextApiResponse } from "next";
import { runAviationApiCommand } from "../../../src/utils/server/aviation_api_bridge";
import { requireApiAuth } from "../../../src/utils/server/require_api_auth";
import {
  enforceRateLimit,
  normalizeOptionalText,
  readSingleQueryValue,
} from "../../../src/utils/server/api_security";

type SessionRequestBody = {
  session_id?: string;
  title?: string;
  pinned?: boolean;
};

function parseLimit(rawValue: string | string[] | undefined, fallback: number) {
  const value = readSingleQueryValue(rawValue);
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.min(Math.max(Math.floor(parsed), 1), 200);
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (
    !enforceRateLimit(req, res, {
      namespace: "chat-session",
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

  try {
    if (req.method === "GET") {
      const search = normalizeOptionalText(req.query.search, 120);
      const filter = normalizeOptionalText(req.query.filter || "all", 20).toLowerCase();
      const limit = parseLimit(req.query.limit, 50);

      if (!["all", "recent", "pinned"].includes(filter)) {
        return res.status(400).json({
          success: false,
          error: "Invalid filter value. Expected all, recent, or pinned.",
        });
      }

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
      const title = normalizeOptionalText(body.title, 120);
      const args = ["session_upsert"];

      if (body.session_id) {
        args.push("--session-id", normalizeOptionalText(body.session_id, 128));
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
