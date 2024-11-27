const cassandra = require('cassandra-driver');
const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
dotenv.config();

async function storeEmbeddings() {
    let client;
    try {
        // Initialize the Cassandra client
        client = new cassandra.Client({
            cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
            credentials: { 
                username: process.env.ASTRA_DB_CLIENT_ID, 
                password: process.env.ASTRA_DB_CLIENT_SECRET 
            },
            keyspace: process.env.ASTRA_DB_KEYSPACE,
        });

        // Connect to Astra DB
        await client.connect();
        console.log('Connected to Astra DB successfully!');
        
        // Load embeddings from file
        const dataPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
        let embeddings;
        try {
            const rawData = fs.readFileSync(dataPath, 'utf-8');
            embeddings = JSON.parse(rawData);

        // Debugging: Print the first embedding for inspection
            console.log("Sample Embedding:", embeddings[0]);
        } catch (err) {
            throw new Error(`Failed to read or parse embeddings file: ${err.message}`);
        }

        // Define the insert query
        const query = `
            INSERT INTO aviation_documents (chunk_id, filename, embedding, metadata)
            VALUES (?, ?, ?, ?);
        `;

        // Insert embeddings into the table
        const batchSize = 100;
        for (let i = 0; i < embeddings.length; i += batchSize) {
            const batch = embeddings.slice(i, i + batchSize).map(embedding => ({
                query: query,
                params: [
                    embedding.chunk_id,
                    embedding.filename,
                    Buffer.from(JSON.stringify(embedding.embedding)),
                    null,
                ]
            }));

            try {
                await client.batch(batch, { prepare: true });
                console.log(`Stored embeddings ${i + 1} to ${Math.min(i + batchSize, embeddings.length)}`);
            } catch (err) {
                console.error(`Failed to store embeddings ${i + 1} to ${Math.min(i + batchSize, embeddings.length)}:`, err);
            }
        }

        console.log('All embeddings stored successfully!');
        console.log(`Processing ${embeddings.length} embeddings...`);

    } catch (err) {
        console.error('Error connecting to Astra DB or processing embeddings:', err);
    } finally {
        if (client) {
            await client.shutdown();
        }
    }
}

// Run the function
storeEmbeddings();