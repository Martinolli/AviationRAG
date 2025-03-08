import { Client } from 'cassandra-driver';
import fs from 'fs';
import fsPromises from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';

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

async function insertEmbeddings() {
    // Check if required environment variables are set
    const requiredEnvVars = ['ASTRA_DB_SECURE_BUNDLE_PATH', 'ASTRA_DB_CLIENT_ID', 'ASTRA_DB_CLIENT_SECRET', 'ASTRA_DB_KEYSPACE'];
    for (const envVar of requiredEnvVars) {
        if (!process.env[envVar]) {
            console.error(`Error: ${envVar} is not set in the environment variables.`);
            process.exit(1);
        }
    }

    const client = new Client({
        cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
        credentials: {
            username: process.env.ASTRA_DB_CLIENT_ID,
            password: process.env.ASTRA_DB_CLIENT_SECRET,
        },
        keyspace: process.env.ASTRA_DB_KEYSPACE,
    });

    try {
        await client.connect();
        console.log('Connected to Astra DB');

        const embeddingsPath = path.join(projectRoot, 'data', 'embeddings', 'aviation_embeddings.json');
        
        console.log(`Attempting to read embeddings from: ${embeddingsPath}`);

        const embeddingsData = JSON.parse(await fsPromises.readFile(embeddingsPath, 'utf8'));

        // Query to get all existing chunk_ids and filenames
        const existingDataQuery = 'SELECT chunk_id, filename FROM aviation_documents';

        const existingChunks = new Set();
        const existingFiles = new Set();
        const result = await client.execute(existingDataQuery);
        result.rows.forEach(row => {
            existingChunks.add(row.chunk_id);
            existingFiles.add(row.filename);
        });

        console.log(`🔍 Retrieved ${existingChunks.size} existing chunk_ids from AstraDB.`);

        const insertQuery = 'INSERT INTO aviation_documents (chunk_id, filename, text, tokens, embedding) VALUES (?, ?, ?, ?, ?)';

        let successCount = 0;
        let skippedCount = 0;
        let errorCount = 0;
        let totalEmbeddings = embeddingsData.length;

        for (const item of embeddingsData) {
            // ✅ Execute query first before logging result
            if (existingChunks.has(item.chunk_id)) {
                skippedCount++;
                continue;
                }    
            
                try {
                // Convert the embedding array to a Buffer
                const embeddingBuffer = Buffer.from(new Float32Array(item.embedding).buffer);

                await client.execute(insertQuery, [
                    item.chunk_id,
                    item.filename,
                    item.text,
                    item.tokens,
                    embeddingBuffer
                ], { prepare: true });
                
                successCount++;
                existingChunks.add(item.chunk_id);
                existingFiles.add(item.filename);

                if (successCount % 10 === 0) {
                    console.log(`Progress: ${successCount + skippedCount} / ${totalEmbeddings}`);
                }

            } catch (insertError) {
                console.error(`Error inserting embedding for chunk_id ${item.chunk_id}:`, insertError);
                errorCount++;
            }
        }

        console.log(`\nInsertion complete.`);
        console.log(`Total embeddings in local file: ${totalEmbeddings}`);
        console.log(`Successful insertions: ${successCount}`);
        console.log(`Skipped (already existing): ${skippedCount}`);
        console.log(`Failed insertions: ${errorCount}`);
        console.log(`Total unique files in database after insertion: ${existingFiles.size}`);

        if (successCount === 0 && skippedCount === totalEmbeddings) {
            console.log("\n⚠️ All embeddings were already in the database. No new insertions were made.");
        } else if (successCount > 0) {
            console.log(`\n✅ Successfully inserted embeddings for ${successCount} new chunks.`);
        }
        
    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
        console.log('Disconnected from Astra DB');
    }
}

insertEmbeddings().catch(err => {
    console.error('Unhandled error in insertEmbeddings:', err);
    process.exit(1);
});