import { Client } from 'cassandra-driver';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';
import fs from 'fs';
import { getAstraCredentials, getMissingAstraEnvVars } from './astra_auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Navigate to the project root (AviationRAG) and load the .env file
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const envPath = path.join(projectRoot, '.env');

if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    console.log(`Loaded .env file from: ${envPath}`);
} else {
    console.error(`Could not find .env file at ${envPath}. Please ensure it exists in the AviationRAG directory.`);
    process.exit(1);
}

// Log environment variables
console.log('Environment variables:');
console.log('ASTRA_DB_SECURE_BUNDLE_PATH:', process.env.ASTRA_DB_SECURE_BUNDLE_PATH);
console.log('ASTRA_DB_APPLICATION_TOKEN:', process.env.ASTRA_DB_APPLICATION_TOKEN ? '[REDACTED]' : 'Not set');
console.log('ASTRA_DB_CLIENT_ID:', process.env.ASTRA_DB_CLIENT_ID ? '[REDACTED]' : 'Not set');
console.log('ASTRA_DB_CLIENT_SECRET:', process.env.ASTRA_DB_CLIENT_SECRET ? '[REDACTED]' : 'Not set');
console.log('ASTRA_DB_KEYSPACE:', process.env.ASTRA_DB_KEYSPACE);

async function checkAstraDBContent() {
    let client;

    try {
        const missing = getMissingAstraEnvVars();
        if (missing.length > 0) {
            console.error(`Missing required env var(s): ${missing.join(', ')}`);
            process.exit(1);
        }

        const { username, password, authMode } = getAstraCredentials();
        console.log(`Using Astra auth mode: ${authMode}`);

        // Initialize the Cassandra client
        client = new Client({
            cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
            credentials: { 
                username,
                password 
            },
            keyspace: process.env.ASTRA_DB_KEYSPACE,
        });

        // Connect to Astra DB
        await client.connect();
        console.log('Connected to Astra DB successfully!');

        // Query to get all rows from the table
        const query = 'SELECT * FROM aviation_documents';
        const result = await client.execute(query, [], { fetchSize: 1000 });

        let allRows = [];
        for await (const row of result) {
            // Convert embedding Buffer to array if it exists
            if (row.embedding && row.embedding instanceof Buffer) {
                const float32Array = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
                row.embedding = Array.from(float32Array);
            }
            allRows.push(row);
        }

        console.log(`Total number of documents retrieved: ${allRows.length}`);

        // Save data to a JSON file
        const outputPath = path.join(projectRoot, 'data', 'astra_db', 'astra_db_content.json');
        fs.writeFileSync(outputPath, JSON.stringify(allRows, null, 2));
        console.log(`Data saved to: ${outputPath}`);

        // Get unique filenames
        const uniqueFilenames = new Set(allRows.map(row => row.filename));
        console.log("\nUnique filenames in the database:");
        uniqueFilenames.forEach(filename => console.log(filename));
        console.log(`\nTotal number of unique files: ${uniqueFilenames.size}`);
        
    } catch (err) {
        console.error('Error querying Astra DB:', err);
    } finally {
        if (client) {
            await client.shutdown();
        }
    }
}

checkAstraDBContent();
