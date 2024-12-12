const cassandra = require('cassandra-driver');
const dotenv = require('dotenv');

dotenv.config();

async function createTable() {
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

        // Check if the table exists
        const checkTableQuery = `
            SELECT table_name 
            FROM system_schema.tables 
            WHERE keyspace_name = '${process.env.ASTRA_DB_KEYSPACE}' 
            AND table_name = 'aviation_documents'
        `;
        const result = await client.execute(checkTableQuery);

        if (result.rows.length > 0) {
            // Table exists, truncate it
            const truncateQuery = `TRUNCATE TABLE aviation_documents`;
            await client.execute(truncateQuery);
            console.log('Existing table truncated');
        } else {
            console.log('Table does not exist, will create new');
        }

        const createQuery = `
            CREATE TABLE IF NOT EXISTS aviation_documents (
                chunk_id TEXT PRIMARY KEY,
                filename TEXT,
                text TEXT,
                tokens INT,
                embedding BLOB
            )
        `;

        await client.execute(createQuery);
        console.log('Table created or verified successfully');
    } catch (err) {
        console.error('Error:', err);
    } finally {
        await client.shutdown();
    }
}

createTable();