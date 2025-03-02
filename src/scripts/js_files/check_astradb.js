const cassandra = require('cassandra-driver');
const dotenv = require('dotenv');
dotenv.config();

const checkContent = async () => {
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

        // Query to fetch specific fields
        const query = `SELECT filename, chunk_id FROM aviation_data.aviation_documents;`;
        const result = await client.execute(query);

        console.log("Fetched Results:");
        result.rows.forEach(row => {
            console.log(`Filename: ${row.filename}, Chunk ID: ${row.chunk_id}`);
        });
    } catch (err) {
        console.error("Error querying the database:", err);
    } finally {
        await client.shutdown();
    }
};

checkContent();
