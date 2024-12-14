const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');

// Load environment variables
dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// Utility function to add delay
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to process a single chunk
async function processChunk(chunk, filename, index) {
  let attempts = 0;
  const maxAttempts = 3;
  const chunk_id = `${filename}-${index}`; // Generate chunk ID dynamically

  while (attempts < maxAttempts) {
    try {
      const response = await openai.createEmbedding({
        model: 'text-embedding-ada-002',
        input: chunk.text,
      });

      const embeddingVector = response.data.data[0].embedding;
      console.log(`Generated embedding for chunk ID: ${chunk_id}`);
      return {
        chunk_id: chunk_id,
        filename: filename,
        text: chunk.text,
        tokens: chunk.tokens, // Add tokens count for reference
        embedding: embeddingVector,
      };
    } catch (err) {
      attempts++;
      console.error(`Error generating embedding for chunk ID: ${chunk_id} (Attempt ${attempts})`, err);
      if (attempts >= maxAttempts) {
        console.error(`Failed to generate embedding for chunk ID: ${chunk_id} after ${maxAttempts} attempts`);
        return null;
      }
      await delay(2000); // Wait before retrying
    }
  }
}

// Function to process files from the chunked documents directory
async function processFile(filePath) {
  const rawData = fs.readFileSync(filePath, 'utf-8');
  const chunkedDoc = JSON.parse(rawData);
  const filename = chunkedDoc.filename;
  const category = chunkedDoc.category; // Metadata
  const embeddings = [];

  console.log(`Processing file: ${filename}, Category: ${category}`);
  for (let i = 0; i < chunkedDoc.chunks.length; i++) {
    const chunk = chunkedDoc.chunks[i];
    const result = await processChunk(chunk, filename, i);
    if (result) embeddings.push(result);
    await delay(500); // Delay between processing chunks
  }

  return embeddings;
}

// Main function to generate embeddings
async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.join(__dirname, '../../data/processed/chunked_documents');
    const outputPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
    const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith('.json'));
    let allEmbeddings = [];

    console.log(`Found ${files.length} files to process.`);
    for (const file of files) {
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