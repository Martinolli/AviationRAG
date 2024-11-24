// Import necessary libraries
const { createClient } = require('@astrajs/collections');
const dotenv = require('dotenv');

// Load environment variables from .env
dotenv.config();

// Create connection configuration
const clientConfig = {
    astraDatabaseId: process.env.ASTRA_DB_ID,
    astraDatabaseRegion: process.env.ASTRA_DB_REGION,
    username: process.env.ASTRA_DB_CLIENT_ID,
    password: process.env.ASTRA_DB_CLIENT_SECRET,
    secureBundlePath: process.env.ASTRA_DB_SECURE_BUNDLE_PATH,
    keyspace: process.env.ASTRA_DB_KEYSPACE,
};

// Function to connect to Astra DB and test the connection
async function connectToAstra() {
    try {
        // Create the Astra DB client
        const client = await createClient(clientConfig);
        console.log('Successfully connected to Astra DB!');
    } catch (err) {
        console.error('Failed to connect to Astra DB:', err);
    }
}

connectToAstra();

