const cassandra = require('cassandra-driver');
const dotenv = require('dotenv');
dotenv.config();

async function connectToAstra() {
  try {
    // Initialize the Cassandra client with credentials
    const client = new cassandra.Client({
      cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
      credentials: {
        username: process.env.ASTRA_DB_CLIENT_ID,
        password: process.env.ASTRA_DB_CLIENT_SECRET,
      },
      keyspace: process.env.ASTRA_DB_KEYSPACE,
    });

    // Test the connection
    await client.connect();
    console.log('Successfully connected to Astra DB!');

    // Optionally, execute a simple query
    const query = 'SELECT release_version FROM system.local';
    const result = await client.execute(query);
    console.log('Cassandra release version:', result.rows[0].release_version);

    // Close the connection
    await client.shutdown();
  } catch (err) {
    console.error('Failed to connect to Astra DB:', err);
  }
}

connectToAstra();
