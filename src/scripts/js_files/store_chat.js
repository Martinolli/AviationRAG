import { Client } from "cassandra-driver";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import path from "path";
import winston from "winston";
import { format } from "date-fns";
import fs from "fs";
import { getAstraCredentials, getMissingAstraEnvVars } from "./astra_auth.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..", "..");
const envPath = path.resolve(projectRoot, ".env");

dotenv.config({ path: envPath });

const logDir = path.resolve(projectRoot, "logs");
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

const logFileName = `store_chat_${format(new Date(), "yyyy-MM-dd")}.log`;
const logFilePath = path.join(logDir, logFileName);

const logger = winston.createLogger({
  level: "info",
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message }) => `${timestamp} [${level}]: ${message}`)
  ),
  transports: [
    new winston.transports.File({
      filename: logFilePath,
      maxsize: 5242880,
      maxFiles: 5,
      tailable: true,
    }),
  ],
});

logger.info(`Logging to: ${logFilePath}`);

const missing = getMissingAstraEnvVars();
if (missing.length > 0) {
  logger.error(`Missing required env var(s): ${missing.join(", ")}`);
  process.exit(1);
}

const { username, password, authMode } = getAstraCredentials();
logger.info(`Using Astra auth mode: ${authMode}`);

if (!process.argv[2]) {
  logger.error("No JSON payload argument received.");
  process.exit(1);
}

let chatData;
try {
  chatData = JSON.parse(process.argv[2]);
} catch (error) {
  logger.error(`Invalid JSON payload: ${error.message}`);
  process.exit(1);
}

const sessionId = String(chatData.session_id || "").trim();
const userQuery = chatData.user_query ? String(chatData.user_query).replace(/[\r\n]+/g, " ").trim() : "";
const aiResponse = chatData.ai_response ? String(chatData.ai_response).replace(/[\r\n]+/g, " ").trim() : "";
const limitValue = Math.min(Math.max(Number(chatData.limit) || 5, 1), 50);

if (!sessionId) {
  logger.error("Invalid session_id");
  process.exit(1);
}

const client = new Client({
  cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
  credentials: {
    username,
    password,
  },
  keyspace: process.env.ASTRA_DB_KEYSPACE,
});

async function handleStore() {
  if (!userQuery) {
    throw new Error("Invalid user_query");
  }
  if (!aiResponse) {
    throw new Error("Invalid ai_response");
  }

  const query = `
    INSERT INTO aviation_conversation_history (session_id, timestamp, user_query, ai_response)
    VALUES (?, toTimestamp(now()), ?, ?)
  `;

  await client.execute(query, [sessionId, userQuery, aiResponse], { prepare: true });
  console.log(JSON.stringify({ success: true, action: "store" }));
}

async function handleRetrieve() {
  const query = `
    SELECT session_id, timestamp, user_query, ai_response
    FROM aviation_conversation_history
    WHERE session_id = ?
    ORDER BY timestamp DESC
    LIMIT ${limitValue};
  `;

  const result = await client.execute(query, [sessionId], { prepare: true });
  const formattedData = result.rows.map((row) => ({
    session_id: row.session_id,
    timestamp: row.timestamp,
    user_query: row.user_query,
    ai_response: row.ai_response,
  }));

  logger.info(`Retrieved ${formattedData.length} chat messages for session: ${sessionId}`);
  console.log(JSON.stringify({ success: true, messages: formattedData }));
}

async function handleDelete() {
  const countQuery = `
    SELECT COUNT(*) AS total
    FROM aviation_conversation_history
    WHERE session_id = ?;
  `;
  const countResult = await client.execute(countQuery, [sessionId], { prepare: true });
  const totalValue = countResult.rows?.[0]?.total;
  const deletedRows = Number(
    typeof totalValue === "number" ? totalValue : (totalValue?.toString?.() || "0")
  );

  const deleteQuery = `
    DELETE FROM aviation_conversation_history
    WHERE session_id = ?;
  `;
  await client.execute(deleteQuery, [sessionId], { prepare: true });

  logger.info(`Deleted chat history for session ${sessionId}. Rows removed: ${deletedRows}`);
  console.log(JSON.stringify({ success: true, action: "delete", deleted_rows: deletedRows }));
}

async function main() {
  try {
    await client.connect();
    logger.info("Connected to Astra DB successfully.");

    if (chatData.action === "store") {
      await handleStore();
      return;
    }

    if (chatData.action === "retrieve") {
      await handleRetrieve();
      return;
    }

    if (chatData.action === "delete") {
      await handleDelete();
      return;
    }

    logger.error(`Invalid action specified: ${chatData.action}`);
    process.exit(1);
  } catch (error) {
    logger.error(`Error handling chat action: ${error.message}`);
    if (error.info) {
      try {
        logger.error(`Error info: ${JSON.stringify(error.info)}`);
      } catch {
        logger.error("Error info: [unserializable]");
      }
    }
    if (error.code) {
      logger.error(`Error code: ${error.code}`);
    }
    process.exit(1);
  } finally {
    await client.shutdown();
  }
}

main();
