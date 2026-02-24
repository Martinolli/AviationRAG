import cassandra from 'cassandra-driver';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { getAstraCredentials, getMissingAstraEnvVars } from './astra_auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..', '..');
dotenv.config({ path: path.join(projectRoot, '.env') });

async function createTable() {
    const missing = getMissingAstraEnvVars();
    if (missing.length > 0) {
        console.error(`Missing required env var(s): ${missing.join(', ')}`);
        process.exit(1);
    }

    const { username, password, authMode } = getAstraCredentials();
    console.log(`Using Astra auth mode: ${authMode}`);

    const client = new cassandra.Client({
        cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
        credentials: {
            username,
            password,
        },
        keyspace: process.env.ASTRA_DB_KEYSPACE,
    });

    try {
        await client.connect();
        console.log('Connected to Astra DB');

        // Check if the table exists
        const checkTableQuery = `
            SELECT table_name 
            FROM system_schema.tables 
            WHERE keyspace_name = '${process.env.ASTRA_DB_KEYSPACE}' 
            AND table_name = 'aviation_documents'
        `;
        const result = await client.execute(checkTableQuery);

        if (result.rows.length > 0) {
            // Table exists, truncate it
            const truncateQuery = `TRUNCATE TABLE aviation_documents`;
            await client.execute(truncateQuery);
            console.log('Existing table truncated');
        } else {
            console.log('Table does not exist, will create new');
        }

        const createQuery = `
            CREATE TABLE IF NOT EXISTS aviation_documents (
                chunk_id TEXT PRIMARY KEY,
                filename TEXT,
                text TEXT,
                tokens INT,
                embedding BLOB
            )
        `;

        await client.execute(createQuery);
        console.log('Table created or verified successfully');
    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
    }
}

createTable();
