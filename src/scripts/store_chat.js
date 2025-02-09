import { Client } from 'cassandra-driver';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';

// Resolve environment variables
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

// Log environment variables for debugging
console.log('ASTRA_DB_SECURE_BUNDLE_PATH:', process.env.ASTRA_DB_SECURE_BUNDLE_PATH);
console.log('ASTRA_DB_CLIENT_ID:', process.env.ASTRA_DB_CLIENT_ID);
console.log('ASTRA_DB_CLIENT_SECRET:', process.env.ASTRA_DB_CLIENT_SECRET);
console.log('ASTRA_DB_KEYSPACE:', process.env.ASTRA_DB_KEYSPACE);

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
        console.log('Connected to Astra DB successfully!');
        
        // Log the data we're trying to insert
        console.log('Attempting to insert data:', { session_id, user_query, ai_response });

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
            console.log("Storing chat message...");
            const query = `
                INSERT INTO aviation_conversation_history (session_id, timestamp, user_query, ai_response)
                VALUES (:session_id, toTimestamp(now()), :user_query, :ai_response)
            `;

            const params = { session_id, user_query, ai_response };

            await client.execute(query, params, { prepare: true });

            console.log("Chat stored successfully!");
            
        } else if (chatData.action === "retrieve") {
            // retrieve chat messages
            console.log("Retrieving chat messages...");

            const query = `
                SELECT session_id, timestamp, user_query, ai_response
                FROM aviation_conversation_history
                WHERE session_id = :session_id
                ORDER BY timestamp DESC
                LIMIT :limit;
            `;

            const params = { session_id: chatData.session_id, limit: chatData.limit || 5 };
            const result = await client.execute(query, params, { prepare: true });

            console.log("Chat history retrieved:", result.rows);
            console.log(JSON.stringify(result.rows));  // Output JSON for Python to parse

        } else {
            console.error("Invalid action specified in chatData");
            process.exit(1);  // Exit with non-zero status
        }

    } catch (error) {
        console.error("Error retrieving chat:", error);
        if (error.info) console.error("Error info:", error.info);
        if (error.code) console.error("Error code:", error.code);
        if (error.query) console.error("Failed query:", error.query);
        process.exit(1);  // Ensure the script exits with a non-zero status
    } finally {
        await client.shutdown();
    }
}

storeChat();