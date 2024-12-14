const cassandra = require('cassandra-driver');
require('dotenv').config();

async function checkAstraDBContent() {
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

        // Query to get a sample of rows from the table
        const sampleQuery = 'SELECT * FROM aviation_documents LIMIT 5';
        const sampleResult = await client.execute(sampleQuery);

        console.log("Sample data from AstraDB:");
        sampleResult.rows.forEach((row, index) => {
            console.log(`\nDocument ${index + 1}:`);
            console.log(`Filename: ${row.filename}`);
            console.log(`Chunk ID: ${row.chunk_id}`);
            console.log(`Embedding length: ${row.embedding ? row.embedding.length : 'N/A'}`);
        });

        // Get the count of rows in the table
        const countQuery = 'SELECT COUNT(*) FROM aviation_documents';
        const countResult = await client.execute(countQuery);
        console.log(`\nTotal number of documents in AstraDB: ${countResult.rows[0].count.toString()}`);

    } catch (err) {
        console.error('Error querying Astra DB:', err);
    } finally {
        if (client) {
            await client.shutdown();
        }
    }
}

checkAstraDBContent();