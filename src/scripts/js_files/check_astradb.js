import { Client } from "cassandra-driver";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import fs from "fs";

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
  const client = new Client({
    cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
    credentials: {
      username: process.env.ASTRA_DB_CLIENT_ID,
      password: process.env.ASTRA_DB_CLIENT_SECRET,
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
