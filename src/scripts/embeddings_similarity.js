const cassandra = require('cassandra-driver');
const fs = require('fs');
const path = require('path');
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
        })
    ;
    try {
        const result = await client.execute(`
            SELECT * FROM aviation_data.aviation_documents
            WHERE similarity(embedding, ?) > 0.8 LIMIT 5;
        `, [queryVector])
    ;

        console.log("Retrieved Documents:", result.rows);
    } catch (err) {
        console.error("Error retrieving embeddings:", err);
    } finally {
        await client.shutdown();
    }
};

retrieveEmbeddings(["aircraft safety"]);
