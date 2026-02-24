import { Client } from "cassandra-driver";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";
import { getAstraCredentials, getMissingAstraEnvVars } from "./astra_auth.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..", "..");
const envPath = path.join(projectRoot, ".env");

if (fs.existsSync(envPath)) {
  dotenv.config({ path: envPath });
} else {
  dotenv.config();
}

async function checkContent() {
  const missing = getMissingAstraEnvVars();
  if (missing.length > 0) {
    console.error(`Missing required env var(s): ${missing.join(", ")}`);
    process.exit(1);
  }

  const { username, password, authMode } = getAstraCredentials();
  console.log(`Using Astra auth mode: ${authMode}`);

  const client = new Client({
    cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
    credentials: {
      username,
      password,
    },
    keyspace: process.env.ASTRA_DB_KEYSPACE,
  });

  try {
    await client.connect();

    const query = "SELECT filename, chunk_id FROM aviation_documents LIMIT 500";
    const result = await client.execute(query);

    console.log("Fetched Results:");
    result.rows.forEach((row) => {
      console.log(`Filename: ${row.filename}, Chunk ID: ${row.chunk_id}`);
    });
  } catch (error) {
    console.error("Error querying the database:", error);
  } finally {
    await client.shutdown();
  }
}

checkContent().catch((error) => {
  console.error("Unexpected error in check_astradb:", error);
  process.exit(1);
});
