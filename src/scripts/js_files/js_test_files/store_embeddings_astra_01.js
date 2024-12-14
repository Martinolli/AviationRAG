const cassandra = require('cassandra-driver');
const fs = require('fs').promises;
const path = require('path');
const dotenv = require('dotenv');

dotenv.config();

async function insertEmbeddings() {
    const client = new cassandra.Client({
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

        const embeddingsPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
        const embeddingsData = JSON.parse(await fs.readFile(embeddingsPath, 'utf8'));

        const query = 'INSERT INTO aviation_documents (chunk_id, filename, text, tokens, embedding) VALUES (?, ?, ?, ?, ?)';

        for (const item of embeddingsData) {
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
        }

        console.log('All embeddings inserted successfully');
    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
    }
}

insertEmbeddings();