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

        // Drop the existing table (if needed)
        const dropTableQuery = `DROP TABLE IF EXISTS aviation_documents;`;
        await client.execute(dropTableQuery);
        console.log('Old table dropped successfully!');

        // Create the new table with the updated schema
        const createTableQuery = `
            CREATE TABLE IF NOT EXISTS aviation_documents (
                chunk_id TEXT PRIMARY KEY,  -- Use chunk_id as primary key
                filename TEXT,              -- Store the source filename
                embedding BLOB,             -- Store vector embeddings
                metadata TEXT               -- Optional field for future use
            );
        `;
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
