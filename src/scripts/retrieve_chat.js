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

// Parse arguments from command line
const session_id = process.argv[2];
const limit = parseInt(process.argv[3], 10) || 10;

async function retrieveChat() {
    try {
        await client.connect();
        console.log('Connected to Astra DB successfully!');

        const query = `
            SELECT session_id, timestamp, user_query, ai_response
            FROM aviation_data.aviation_conversation_history
            LIMIT ${limit}
        `;

        const params = { session_id, limit };
        const result = await client.execute(query, params, { prepare: true });

        console.log("Chat history retrieved:", result.rows);
        console.log(JSON.stringify(result.rows));  // Output JSON for further inspection

        // Print the last two chat messages
        if (result.rows.length > 0) {
            console.log("Last two chat messages:");
            const lastTwoMessages = result.rows.slice(-2);
            lastTwoMessages.forEach((row, index) => {
                console.log(`Message ${index + 1}:`);
                console.log(`Session ID: ${row.session_id}`);
                console.log(`Timestamp: ${row.timestamp}`);
                console.log(`User Query: ${row.user_query}`);
                console.log(`AI Response: ${row.ai_response}`);
            });
        } else {
            console.log("No chat messages found for the given session_id.");
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

retrieveChat();