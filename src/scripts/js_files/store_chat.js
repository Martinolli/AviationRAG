import { Client } from 'cassandra-driver';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';
import winston from 'winston';
import {format} from 'date-fns';
import fs from 'fs';


// Resolve environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../../../.env') });
console.log("✅ Environment variables loaded:", process.env.ASTRA_DB_CLIENT_ID ? "✅" : "❌ MISSING");

// Set up logging
const logDir = path.resolve(__dirname, '../../../logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// Add this line for debugging
console.log(`Log directory: ${logDir}`);

const logFileName = `store_chat_${format(new Date(), 'yyyy-MM-dd')}.log`;
const logFilePath = path.join(logDir, logFileName);

const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.printf(({ timestamp, level, message }) => {
        return `${timestamp} [${level}]: ${message}`;
      })
    ),
    transports: [
      new winston.transports.Console(),
      new winston.transports.File({ 
        filename: logFilePath,
        maxsize: 5242880, // 5MB
        maxFiles: 5,
        tailable: true
      })
    ]
  });

logger.info(`Logging to: ${logFilePath}`);  // Add this for debugging

// Log environment variables for debugging
// logger.info('ASTRA_DB_SECURE_BUNDLE_PATH:', process.env.ASTRA_DB_SECURE_BUNDLE_PATH);
// logger.info('ASTRA_DB_CLIENT_ID:', process.env.ASTRA_DB_CLIENT_ID);
// logger.info('ASTRA_DB_CLIENT_SECRET:', process.env.ASTRA_DB_CLIENT_SECRET);
// logger.info('ASTRA_DB_KEYSPACE:', process.env.ASTRA_DB_KEYSPACE);

// Initialize the Cassandra client
const client = new Client({
    cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
    credentials: { 
        username: process.env.ASTRA_DB_CLIENT_ID, 
        password: process.env.ASTRA_DB_CLIENT_SECRET 
    },
    keyspace: process.env.ASTRA_DB_KEYSPACE,
});

// Parse arguments from Python
const chatData = JSON.parse(process.argv[2]);

// Ensure session_id is a string and sanitize user inputs
const session_id = String(chatData.session_id).trim();  
const user_query = chatData.user_query ? chatData.user_query.replace(/[\r\n]+/g, ' ').trim() : '';  
const ai_response = chatData.ai_response ? chatData.ai_response.replace(/[\r\n]+/g, ' ').trim() : '';

async function storeChat() {
    try {
        await client.connect();
        logger.info('Connected to Astra DB successfully!');
        
        // Log the data we're trying to insert
        logger.info(`Storing chat message for session: ${session_id}`);

        // Additional validation
        if (typeof session_id !== 'string' || session_id.length === 0) {
            throw new Error('Invalid session_id');
        }

        if (chatData.action === "store") {
            if (typeof user_query !== 'string' || user_query.length === 0) {
                throw new Error('Invalid user_query');
            }
            if (typeof ai_response !== 'string' || ai_response.length === 0) {
                throw new Error('Invalid ai_response');
            }

            // store chat message
            // logger.info("Storing chat message...");
            const query = `
                INSERT INTO aviation_conversation_history (session_id, timestamp, user_query, ai_response)
                VALUES (:session_id, toTimestamp(now()), :user_query, :ai_response)
            `;

            const params = { session_id, user_query, ai_response };

            await client.execute(query, params, { prepare: true });

            // logger.info("Chat stored successfully!");

            // Log chat details to separate file
            // logger.info(`Stored chat for session: ${session_id}`);
            
        } else if (chatData.action === "retrieve") {
            // retrieve chat messages
            // logger.info(`Retrieving chat messages for session_id: ${chatData.session_id}`);

            const limitValue = chatData.limit || 5;

            const query = `
                SELECT session_id, timestamp, user_query, ai_response
                FROM aviation_data.aviation_conversation_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT 10;
            `;

            const params = [chatData.session_id];

            const result = await client.execute(query, params, { prepare: true });

            logger.info(`Retrieved ${result.rows.length} chat messages for session: ${session_id}`);
            const responsePayload = { success: true, data: result.rows };

            const formattedData = result.rows.map(row => ({
                session_id: row.session_id,
                timestamp: row.timestamp,
                user_query: row.user_query,
                ai_response: row.ai_response
            }));
            
            console.log(JSON.stringify({ success: true, messages: formattedData }));
            
            return responsePayload;


        } else {
            logging.error("Invalid action specified in chatData");
            process.exit(1);  // Exit with non-zero status
        }

    } catch (error) {
        logger.error("Error retrieving chat:", error);
        if (error.info) logger.error("Error info:", error.info);
        if (error.code) logger.error("Error code:", error.code);
        if (error.query) logger.error("Failed query:", error.query);

        process.exit(1);  // Ensure the script exits with a non-zero status
    } finally {
        await client.shutdown();
    }
}

storeChat();