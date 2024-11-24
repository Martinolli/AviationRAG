const fs = require('fs');
const path = require('path');
const { createClient } = require('@astrajs/collections');
const dotenv = require('dotenv');

// Load environment variables
dotenv.config();

async function storeEmbeddings() {
  try {
    // Connect to Astra DB
    const client = await createClient({
      astraDatabaseId: process.env.ASTRA_DB_ID,
      astraDatabaseRegion: process.env.ASTRA_DB_REGION,
      astraApplicationToken: process.env.ASTRA_DB_TOKEN,
    });

    const table = client.namespace(process.env.ASTRA_DB_KEYSPACE).collection('aviation_documents');

    // Load embeddings
    const dataPath = path.join(__dirname, '../../data/processed/aviation_embeddings.json');
    const rawData = fs.readFileSync(dataPath, 'utf-8');
    const embeddings = JSON.parse(rawData);

    // Insert each embedding into the table
    for (const embedding of embeddings) {
      const id = embedding.id; // Unique ID
      const payload = {
        title: embedding.title,
        text_chunk: embedding.text_chunk,
        embedding: embedding.embedding, // Vector embedding
      };

      await table.create(id, payload);
      console.log(`Stored embedding for ID: ${id}`);
    }

    console.log('All embeddings stored successfully!');
  } catch (err) {
    console.error('Error storing embeddings:', err);
  }
}

// Run the function
storeEmbeddings();
