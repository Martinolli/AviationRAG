const cassandra = require('cassandra-driver');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

async function storeEmbeddings() {
    try {
        // Initialize the Cassandra client
        const client = new cassandra.Client({
            cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
            keyspace: process.env.ASTRA_DB_KEYSPACE,
        });

        // Connect to Astra DB
        await client.connect();
        console.log('Connected to Astra DB successfully!');

        // Load embeddings from file
        const dataPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
        const rawData = fs.readFileSync(dataPath, 'utf-8');
        const embeddings = JSON.parse(rawData);

        // Insert embeddings into the table
        for (const embedding of embeddings) {
            const query = `
                INSERT INTO aviation_documents (chunk_id, filename, embedding, metadata)
                VALUES (?, ?, ?, ?);
            `;

            const params = [
                embedding.chunk_id,                         // Chunk ID
                embedding.filename,                        // Filename
                Buffer.from(JSON.stringify(embedding.embedding)), // Convert embedding to blob
                null,                                      // Metadata (optional)
            ];

            try {
                await client.execute(query, params, { prepare: true });
                console.log(`Stored embedding for Chunk ID: ${embedding.chunk_id}`);
            } catch (err) {
                console.error(`Failed to store embedding for Chunk ID: ${embedding.chunk_id}`, err);
            }
        }

        console.log('All embeddings stored successfully!');
        await client.shutdown();
    } catch (err) {
        console.error('Error connecting to Astra DB or processing embeddings:', err);
    }
}

// Run the function
storeEmbeddings();