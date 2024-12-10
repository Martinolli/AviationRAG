const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');

// Load environment variables from .env
dotenv.config();

// Initialize OpenAI client using the OpenAI SDK
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// Function to add delay
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to process a single chunk
async function processChunk(chunk, filename) {
  let attempts = 0;
  const maxAttempts = 3;

  while (attempts < maxAttempts) {
    try {
      const response = await openai.createEmbedding({
        model: 'text-embedding-ada-002',
        input: chunk.text,
      });

      const embeddingVector = response.data.data[0].embedding;
      console.log(`Generated embedding for chunk ID: ${chunk.chunk_id}`);
      return {
        chunk_id: chunk.chunk_id,
        filename: filename,
        text: chunk.text,
        embedding: embeddingVector,
      };
    } catch (err) {
      attempts++;
      console.error(`Error generating embedding for chunk ID: ${chunk.chunk_id} (Attempt ${attempts})`, err);
      if (attempts >= maxAttempts) {
        console.error(`Failed to generate embedding for chunk ID: ${chunk.chunk_id} after ${maxAttempts} attempts`);
        return null;
      }
      await delay(2000); // Delay before retrying
    }
  }
}

// Function to process a single file
async function processFile(filePath) {
  const rawData = fs.readFileSync(filePath, 'utf-8');
  const chunkedDocs = JSON.parse(rawData);
  const filename = chunkedDocs.filename;
  const embeddings = [];

  for (const chunk of chunkedDocs.chunks) {
    const result = await processChunk(chunk, filename);
    if (result) embeddings.push(result);
    await delay(500); // Adjust delay as needed
  }

  return embeddings;
}

// Main function
async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.join(__dirname, '../../data/processed/chunked_documents');
    const outputPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
    const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith('.json'));
    let allEmbeddings = [];

    console.log(`Found ${files.length} files to process.`);
    for (const file of files) {
      console.log(`Processing file: ${file}`);
      const filePath = path.join(chunkedDocsPath, file);
      const embeddings = await processFile(filePath);
      allEmbeddings = allEmbeddings.concat(embeddings);
    }

    // Save all embeddings to a JSON file
    await fs.promises.writeFile(outputPath, JSON.stringify(allEmbeddings, null, 2));
    console.log(`Embeddings saved to ${outputPath}`);
  } catch (err) {
    console.error('Error while generating embeddings:', err);
  }
}

// Run the function
generateEmbeddings();
