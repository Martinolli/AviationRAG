const cassandra = require('cassandra-driver');
const dotenv = require('dotenv');
dotenv.config();

const retrieveEmbeddings = async (queryVector) => {
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

        // Ensure the client is connected
        await client.connect();

        // Query the table using similarity (if available)
        const query = `
            SELECT * FROM aviation_data.aviation_documents
            WHERE similarity(embedding, ?) > 0.8 LIMIT 5;
        `;
        const params = [queryVector];

        const result = await client.execute(query, params, { prepare: true });

        console.log("Retrieved Documents:", result.rows);
    } catch (err) {
        console.error("Error retrieving embeddings:", err);
    } finally {
        if (client) {
            await client.shutdown();
        }
    }
};

// Example usage with a sample query vector
const queryVector = [0.123, 0.456, 0.789, 0.101, 0.112]; // Replace with actual embedding
retrieveEmbeddings(queryVector);
