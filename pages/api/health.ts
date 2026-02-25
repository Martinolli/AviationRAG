import type { NextApiRequest, NextApiResponse } from "next";
import path from "path";
import fs from "fs";

type HealthResponse = {
  success: boolean;
  service: string;
  timestamp: string;
  checks: {
    openai_api_key: boolean;
    astra_token: boolean;
    astra_bundle_path_set: boolean;
    astra_keyspace: boolean;
    python_bridge_script_exists: boolean;
  };
};

export default function handler(
  req: NextApiRequest,
  res: NextApiResponse<HealthResponse>,
) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({
      success: false,
      service: "aviationrag-api",
      timestamp: new Date().toISOString(),
      checks: {
        openai_api_key: false,
        astra_token: false,
        astra_bundle_path_set: false,
        astra_keyspace: false,
        python_bridge_script_exists: false,
      },
    });
  }

  const scriptPath = path.join(
    process.cwd(),
    "src",
    "scripts",
    "py_files",
    "aviationai_api.py",
  );

  return res.status(200).json({
    success: true,
    service: "aviationrag-api",
    timestamp: new Date().toISOString(),
    checks: {
      openai_api_key: Boolean(process.env.OPENAI_API_KEY),
      astra_token: Boolean(process.env.ASTRA_DB_APPLICATION_TOKEN),
      astra_bundle_path_set: Boolean(process.env.ASTRA_DB_SECURE_BUNDLE_PATH),
      astra_keyspace: Boolean(process.env.ASTRA_DB_KEYSPACE),
      python_bridge_script_exists: fs.existsSync(scriptPath),
    },
  });
}

