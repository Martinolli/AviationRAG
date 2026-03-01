import { Client } from 'cassandra-driver';
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { getAstraCredentials, getMissingAstraEnvVars } from './astra_auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables
const envPath = path.resolve(__dirname, '../../../.env');
console.log('Loading .env file from:', envPath);
dotenv.config({ path: envPath });

// Debug: Log all environment variables
console.log('Environment variables:');
console.log('ASTRA_DB_SECURE_BUNDLE_PATH:', process.env.ASTRA_DB_SECURE_BUNDLE_PATH);
console.log('ASTRA_DB_APPLICATION_TOKEN:', process.env.ASTRA_DB_APPLICATION_TOKEN ? '[REDACTED]' : 'Not set');
console.log('ASTRA_DB_KEYSPACE:', process.env.ASTRA_DB_KEYSPACE);

async function checkConsistency() {
    const missing = getMissingAstraEnvVars();
    if (missing.length > 0) {
        console.error(`Missing required env var(s): ${missing.join(', ')}`);
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
        console.log('Connected to Astra DB');

        // Read local embeddings
        const embeddingsPath = path.resolve(__dirname, '../../../data/embeddings/aviation_embeddings.json');
        console.log('Reading embeddings from:', embeddingsPath);
        const localEmbeddings = JSON.parse(await fs.readFile(embeddingsPath, 'utf8'));
        
        // Query Astra DB with explicit paging (single execute() can return only one page).
        const query = 'SELECT chunk_id, embedding FROM aviation_documents';
        const dbRows = [];
        let pageState = null;
        do {
            // eslint-disable-next-line no-await-in-loop
            const page = await client.execute(query, [], { fetchSize: 200, pageState });
            dbRows.push(...page.rows);
            pageState = page.pageState;
        } while (pageState);

        console.log(`Local embeddings count: ${localEmbeddings.length}`);
        console.log(`Astra DB embeddings count: ${dbRows.length}`);

        let matchCount = 0;
        let mismatchCount = 0;
        const dbByChunkId = new Map(dbRows.map((row) => [row.chunk_id, row]));

        for (const localItem of localEmbeddings) {
            const dbItem = dbByChunkId.get(localItem.chunk_id);
            
            if (dbItem) {
                const localEmbedding = new Float32Array(localItem.embedding);
                const dbBuffer = Buffer.isBuffer(dbItem.embedding)
                    ? dbItem.embedding
                    : Buffer.from(dbItem.embedding);
                const dbEmbedding = new Float32Array(
                    dbBuffer.buffer,
                    dbBuffer.byteOffset,
                    dbBuffer.byteLength / Float32Array.BYTES_PER_ELEMENT
                );

                if (compareEmbeddings(localEmbedding, dbEmbedding)) {
                    matchCount++;
                } else {
                    mismatchCount++;
                    console.log(`Mismatch found for chunk_id: ${localItem.chunk_id}`);
                }
            } else {
                mismatchCount++;
                console.log(`No matching DB entry found for chunk_id: ${localItem.chunk_id}`);
            }
        }

        console.log(`Consistency check complete.`);
        console.log(`Matches: ${matchCount}`);
        console.log(`Mismatches: ${mismatchCount}`);

    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
        console.log('Disconnected from Astra DB');
    }
}

function compareEmbeddings(arr1, arr2) {
    if (arr1.length !== arr2.length) return false;
    for (let i = 0; i < arr1.length; i++) {
        if (Math.abs(arr1[i] - arr2[i]) > 1e-6) return false;
    }
    return true;
}

checkConsistency().catch(err => {
    console.error('Unhandled error in checkConsistency:', err);
    process.exit(1);
});
