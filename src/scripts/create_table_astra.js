const cassandra = require('cassandra-driver');
const dotenv = require('dotenv');

// Load environment variables from .env
dotenv.config();

// Define the function to create the table
async function createTable() {
    try {
        // Create an Astra DB client using the secure connect bundle
        const client = new cassandra.Client({
            cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
            credentials: {
                username: process.env.ASTRA_DB_CLIENT_ID,
                password: process.env.ASTRA_DB_CLIENT_SECRET,
            },
            keyspace: process.env.ASTRA_DB_KEYSPACE
        });

        // Connect to the database
        await client.connect();

        // CQL to create the table
        const createTableQuery = `
            CREATE TABLE IF NOT EXISTS aviation_documents (
                id UUID PRIMARY KEY,
                title TEXT,
                text_chunk TEXT,
                embedding BLOB,
                metadata TEXT
            );
        `;

        // Execute the CQL query
        await client.execute(createTableQuery);
        console.log('Table created successfully in Astra DB!');

        // Close the connection
        await client.shutdown();
    } catch (err) {
        console.error('Failed to create table in Astra DB:', err);
    }
}

// Run the function
createTable();
