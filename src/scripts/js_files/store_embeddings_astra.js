import { Client } from 'cassandra-driver';
import fs from 'fs';
import fsPromises from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { getAstraCredentials, getMissingAstraEnvVars } from './astra_auth.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Navigate to the project root (AviationRAG) and load the .env file
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const envPath = path.join(projectRoot, '.env');

const EMBEDDINGS_FILE = path.join(projectRoot, 'data', 'embeddings', 'aviation_embeddings.json');
const ASTRA_CONTENT_FILE = path.join(projectRoot, 'data', 'astra_db', 'astra_db_content.json');

if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    console.log(`Loaded .env file from: ${envPath}`);
} else {
    console.error(`Could not find .env file at ${envPath}. Please ensure it exists in the AviationRAG directory.`);
    process.exit(1);
}

async function insertEmbeddings(newEmbeddings) {
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

        const query = 'INSERT INTO aviation_documents (chunk_id, filename, text, tokens, embedding) VALUES (?, ?, ?, ?, ?)';

        let successCount = 0;
        let errorCount = 0;

        for (const item of newEmbeddings) {
            try {
                // Convert the embedding array to a Buffer
                const embeddingBuffer = Buffer.from(new Float32Array(item.embedding).buffer);

                await client.execute(query, [
                    item.chunk_id,
                    item.filename,
                    item.text,
                    item.tokens,
                    embeddingBuffer
                ], { prepare: true });
                
                console.log(`Inserted embedding for chunk_id: ${item.chunk_id}`);
                successCount++;
            } catch (insertError) {
                console.error(`Error inserting embedding for chunk_id ${item.chunk_id}:`, insertError);
                errorCount++;
            }
        }

        console.log(`Insertion complete. Successful: ${successCount}, Failed: ${errorCount}`);
    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
        console.log('Disconnected from Astra DB');
    }
}

async function main() {
  // 1. Load Local Embeddings
  const localEmbeddings = await loadEmbeddings(EMBEDDINGS_FILE);

  // 2. Load Existing Astra DB Embeddings (from the local JSON)
  const astraEmbeddings = await loadEmbeddings(ASTRA_CONTENT_FILE);

  // 3. Identify New Embeddings (same logic as in check_new_embeddings.js)
  const astraChunkIds = new Set(astraEmbeddings.map(e => e.chunk_id));
  const newEmbeddings = localEmbeddings.filter(embedding => !astraChunkIds.has(embedding.chunk_id));

  // 4. Store only the new embeddings to Astra DB
  if(newEmbeddings.length>0){
    await insertEmbeddings(newEmbeddings);
  } else {
    console.log('No new embeddings to store in Astra DB.');
  }
}

async function loadEmbeddings(filePath) {
  // Same loadEmbeddings as in check_new_embeddings.js
  try {
    if (!fs.existsSync(filePath)) {
      console.log(`File ${filePath} does not exist. Assuming empty database.`);
      return [];
    }
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.log(`Error reading file ${filePath}: ${error.message}`);
    return [];
  }
}
main().catch(console.error);
