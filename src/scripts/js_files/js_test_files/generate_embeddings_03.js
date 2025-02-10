import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';

// Load environment variables
dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Utility function to add delay
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Function to load existing embeddings
function loadExistingEmbeddings(outputPath) {
  if (fs.existsSync(outputPath)) {
    const rawData = fs.readFileSync(outputPath, 'utf-8');
    return JSON.parse(rawData);
  }
  return [];
}

// Function to check if chunk ID exists
function isChunkIdExists(existingEmbeddings, chunkId) {
  console.log("Checking if chunk ID exists:", chunkId);
  console.log("Existing embeddings:", existingEmbeddings);
  return existingEmbeddings.some(embedding => embedding.chunk_id === chunkId);
}

// Function to process a chunk
async function processChunk(chunk, filename, chunkId, metadata, existingEmbeddings) {
  if (isChunkIdExists(existingEmbeddings, chunkId)) {
    console.log(`Skipping duplicate chunk ID: ${chunkId}`);
    return null;
  }

  let attempts = 0;
  const maxAttempts = 3;

  while (attempts < maxAttempts) {
    try {
      const response = await openai.embeddings.create({
        model: 'text-embedding-ada-002',
        input: chunk.text,
      });

      const embeddingVector = response.data[0].embedding;
      console.log(`Generated embedding for chunk ID: ${chunkId}`);
      return {
        chunk_id: chunkId,
        filename,
        metadata,
        text: chunk.text,
        tokens: chunk.tokens,
        embedding: embeddingVector,
      };
    } catch (err) {
      attempts++;
      console.error(`Error generating embedding for chunk ID: ${chunkId} (Attempt ${attempts})`, err);
      if (attempts >= maxAttempts) {
        console.error(`Failed to generate embedding for chunk ID: ${chunkId} after ${maxAttempts} attempts`);
        return null;
      }
      await delay(2000);
    }
  }
}

// Function to process a file
async function processFile(filePath, existingEmbeddings) {
  const embeddings = [];
  const content = fs.readFileSync(filePath, 'utf8');
  const chunkedDoc = JSON.parse(content);
  const { filename, metadata } = chunkedDoc;

  console.log(`Processing file: ${filename}, Metadata:`, metadata);
  for (let i = 0; i < chunkedDoc.chunks.length; i++) {
    const chunk = chunkedDoc.chunks[i];
    const result = await processChunk(chunk, filename, chunk.chunk_id, metadata, existingEmbeddings);
    if (result) embeddings.push(result);
    await delay(500);
  }

  return embeddings;
}

// Function to generate embeddings
async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.resolve('data/processed/chunked_documents');
    const outputPath = path.resolve('data/embeddings/aviation_embeddings.json');
    const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith('.json'));

    if (files.length === 0) {
      console.error('No files found in the chunked_documents directory.');
      return;
    }

    let allEmbeddings = loadExistingEmbeddings(outputPath);

    console.log(`Found ${files.length} files to process.`);
    for (const file of files) {
      const filePath = path.join(chunkedDocsPath, file);
      const embeddings = await processFile(filePath, allEmbeddings);
      allEmbeddings = allEmbeddings.concat(embeddings);
    }

    await fs.promises.writeFile(outputPath, JSON.stringify(allEmbeddings, null, 2));
    console.log(`Embeddings saved to ${outputPath}`);
  } catch (err) {
    console.error('Error while generating embeddings:', err);
  }
}

generateEmbeddings();