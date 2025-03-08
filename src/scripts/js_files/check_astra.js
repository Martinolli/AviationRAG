import { Client } from 'cassandra-driver';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..', '..');

dotenv.config({ path: path.join(projectRoot, '.env') });

async function verifyDBContent() {
    const client = new Client({
        cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
        credentials: { username: process.env.ASTRA_DB_CLIENT_ID, password: process.env.ASTRA_DB_CLIENT_SECRET },
        keyspace: process.env.ASTRA_DB_KEYSPACE,
    });

    try {
        await client.connect();
        console.log('Connected to Astra DB');

        // Count total records
        const countResult = await client.execute('SELECT COUNT(*) FROM aviation_documents');
        console.log(`Total records in aviation_documents: ${countResult.rows[0].count.toString()}`);

        // Sample some records
        const sampleResult = await client.execute('SELECT chunk_id, filename FROM aviation_documents LIMIT 10');
        console.log('Sample records:');
        sampleResult.rows.forEach(row => {
            console.log(`chunk_id: ${row.chunk_id}, filename: ${row.filename}`);
        });

    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
    }
}

verifyDBContent().catch(console.error);