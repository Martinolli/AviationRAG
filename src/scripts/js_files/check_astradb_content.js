import { Client } from 'cassandra-driver';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

async function checkAstraDBContent() {
    let client;

    try {
        // Initialize the Cassandra client
        client = new Client({
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
        const sampleQuery = 'SELECT * FROM aviation_data.aviation_documents LIMIT 5';
        const sampleResult = await client.execute(sampleQuery);

        console.log("Sample data from AstraDB:");
        sampleResult.rows.forEach((row, index) => {
            console.log(`\nDocument ${index + 1}:`);
            console.log(`Filename: ${row.filename}`);
            console.log(`Chunk ID: ${row.chunk_id}`);
            console.log(`Text: ${row.text.substring(0, 100)}...`);
            console.log(`Tokens: ${row.tokens}`);
            if (row.embedding) {
                console.log(`Embedding type: ${typeof row.embedding}`);
                console.log(`Embedding constructor: ${row.embedding.constructor.name}`);
                console.log(`Embedding length: ${row.embedding.length}`);
                if (row.embedding instanceof Buffer) {
                    console.log(`Buffer byte length: ${row.embedding.byteLength}`);
                    const float32Array = new Float32Array(row.embedding.buffer, row.embedding.byteOffset, row.embedding.byteLength / 4);
                    console.log(`Float32Array length: ${float32Array.length}`);
                    console.log(`Embedding sample: ${JSON.stringify(Array.from(float32Array.slice(0, 5)))}`);
                } else {
                    console.log(`Embedding sample: ${JSON.stringify(row.embedding.slice(0, 5))}`);
                }
            } else {
                console.log(`Embedding: N/A`);
            }
        });

        // Get the count of rows in the table
        const countQuery = 'SELECT COUNT(*) FROM aviation_data.aviation_documents';
        const countResult = await client.execute(countQuery);
        console.log(`\nTotal number of documents in AstraDB: ${countResult.rows[0].count.toString()}`);

        // Get the unique filenames
        console.log("\nUnique filenames in the database:");
        const filenamesQuery = 'SELECT filename FROM aviation_data.aviation_documents';
        const filenamesResult = await client.execute(filenamesQuery);
        const uniqueFilenames = new Set(filenamesResult.rows.map(row => row.filename));
        uniqueFilenames.forEach(filename => console.log(filename));

        // Get the number of unique filenames
        console.log(`\nTotal number of unique files: ${uniqueFilenames.size}`);

    } catch (err) {
        console.error('Error querying Astra DB:', err);
    } finally {
        if (client) {
            await client.shutdown();
        }
    }
}

checkAstraDBContent();