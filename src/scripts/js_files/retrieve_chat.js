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
dotenv.config({ path: path.resolve(__dirname, '../../.env') });


// Set up logging
const logDir = path.resolve(__dirname, '../../../logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

// Add this line for debugging
console.log(`Log directory: ${logDir}`);

const logFileName = `retrieve_chat_${format(new Date(), 'yyyy-MM-dd')}.log`;
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
logger.info('ASTRA_DB_SECURE_BUNDLE_PATH:', process.env.ASTRA_DB_SECURE_BUNDLE_PATH);
logger.info('ASTRA_DB_CLIENT_ID:', process.env.ASTRA_DB_CLIENT_ID);
logger.info('ASTRA_DB_CLIENT_SECRET:', process.env.ASTRA_DB_CLIENT_SECRET);
logger.info('ASTRA_DB_KEYSPACE:', process.env.ASTRA_DB_KEYSPACE);

// Initialize the Cassandra client
const client = new Client({
    cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
    credentials: { 
        username: process.env.ASTRA_DB_CLIENT_ID, 
        password: process.env.ASTRA_DB_CLIENT_SECRET 
    },
    keyspace: process.env.ASTRA_DB_KEYSPACE,
});

// Parse arguments from command line
const session_id = process.argv[2];
const limit = parseInt(process.argv[3], 10) || 10;

async function retrieveChat() {
    try {
        await client.connect();
        logger.info('Connected to Astra DB successfully!');

        const query = `
            SELECT session_id, timestamp, user_query, ai_response
            FROM aviation_data.aviation_conversation_history
            LIMIT ${limit}
        `;

        const params = { session_id, limit };
        const result = await client.execute(query, params, { prepare: true });

        logger.info("Chat history retrieved:", result.rows);
        logger.info(JSON.stringify(result.rows));  // Output JSON for further inspection

        // Print the last two chat messages
        if (result.rows.length > 0) {
            console.log("Last two chat messages:");
            const lastTwoMessages = result.rows.slice(-2);
            lastTwoMessages.forEach((row, index) => {
                logger.info(`Message ${index + 1}:`);
                logger.info(`Session ID: ${row.session_id}`);
                logger.info(`Timestamp: ${row.timestamp}`);
                logger.info(`User Query: ${row.user_query}`);
                logger.info(`AI Response: ${row.ai_response}`);
            });
        } else {
            console.log("No chat messages found for the given session_id.");
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

retrieveChat();