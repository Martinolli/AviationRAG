import { Client } from "cassandra-driver";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import { getAstraCredentials, getMissingAstraEnvVars } from "./astra_auth.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..", "..");

dotenv.config({ path: path.join(projectRoot, ".env") });

async function connectToAstra() {
  const missing = getMissingAstraEnvVars();
  if (missing.length > 0) {
    console.error(`Missing required env var(s): ${missing.join(", ")}`);
    process.exit(1);
  }

  const { username, password, authMode } = getAstraCredentials();
  console.log(`Using Astra auth mode: ${authMode}`);

  const client = new Client({
    cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
    credentials: { username, password },
    keyspace: process.env.ASTRA_DB_KEYSPACE,
  });

  try {
    await client.connect();
    console.log("Successfully connected to Astra DB!");

    const query = "SELECT release_version FROM system.local";
    const result = await client.execute(query);
    console.log("Cassandra release version:", result.rows[0].release_version);
  } catch (error) {
    console.error("Failed to connect to Astra DB:", error);
  } finally {
    await client.shutdown();
  }
}

connectToAstra();
