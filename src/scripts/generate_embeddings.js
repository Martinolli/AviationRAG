import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});


function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function loadExistingEmbeddings(outputPath) {
  if (fs.existsSync(outputPath)) {
    const rawData = fs.readFileSync(outputPath, 'utf-8');
    return JSON.parse(rawData);
  }
  return [];
}

function isChunkIdExists(existingEmbeddings, chunk_id) {
  return existingEmbeddings.some(embedding => embedding.chunk_id === chunk_id);
}

async function processChunk(chunk, filename, index, existingEmbeddings) {
  const chunk_id = `${filename}-${index}`;

  if (isChunkIdExists(existingEmbeddings, chunk_id)) {
    console.log(`Skipping duplicate chunk ID: ${chunk_id}`);
    return null;
  }

  let attempts = 0;
  const maxAttempts = 3;

  while (attempts < maxAttempts) {
    try {
      const response = await openai.createEmbedding({
        model: 'text-embedding-ada-002',
        input: chunk.text,
      });

      const embeddingVector = response.data.data[0].embedding;
      console.log(`Generated embedding for chunk ID: ${chunk_id}`);
      return {
        chunk_id,
        filename,
        text: chunk.text,
        tokens: chunk.tokens,
        embedding: embeddingVector,
      };
    } catch (err) {
      attempts++;
      console.error(`Error generating embedding for chunk ID: ${chunk_id} (Attempt ${attempts})`, err);
      if (attempts >= maxAttempts) {
        console.error(`Failed to generate embedding for chunk ID: ${chunk_id} after ${maxAttempts} attempts`);
        return null;
      }
      await delay(2000);
    }
  }
}

async function processFile(filePath, existingEmbeddings) {
  const rawData = fs.readFileSync(filePath, 'utf-8');
  const chunkedDoc = JSON.parse(rawData);
  const filename = chunkedDoc.filename;
  const category = chunkedDoc.category;
  const embeddings = [];

  console.log(`Processing file: ${filename}, Category: ${category}`);
  for (let i = 0; i < chunkedDoc.chunks.length; i++) {
    const chunk = chunkedDoc.chunks[i];
    const result = await processChunk(chunk, filename, i, existingEmbeddings);
    if (result) embeddings.push(result);
    await delay(500);
  }

  return embeddings;
}

async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.resolve(__dirname, '../../data/processed/chunked_documents');
    const outputPath = path.resolve(__dirname, '../../data/embeddings/aviation_embeddings.json');
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
