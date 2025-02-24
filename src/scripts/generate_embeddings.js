import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { createObjectCsvWriter } from 'csv-writer';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const BATCH_SIZE = 5; // Adjust based on API limits and performance
const DELAY_MS = 1000;
const MAX_RETRIES = 3;

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

function isChunkIdExists(existingEmbeddings, chunkId) {
  return existingEmbeddings.some(embedding => embedding.chunk_id === chunkId);
}

async function processChunk(chunk, filename, chunkId, metadata, existingEmbeddings) {
  if (isChunkIdExists(existingEmbeddings, chunkId)) {
    console.log(`Skipping duplicate chunk ID: ${chunkId}`);
    return null;
  }

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
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
      console.error(`Error generating embedding for chunk ID: ${chunkId} (Attempt ${attempt})`, err);
      if (attempt === MAX_RETRIES) {
        console.error(`Failed to generate embedding for chunk ID: ${chunkId} after ${MAX_RETRIES} attempts`);
        return null;
      }
      await delay(DELAY_MS * attempt);
    }
  }
}

async function processChunkBatch(chunks, filename, metadata, existingEmbeddings) {
  const batchPromises = chunks.map(chunk => 
    processChunk(chunk, filename, chunk.chunk_id, metadata, existingEmbeddings)
  );
  return (await Promise.all(batchPromises)).filter(result => result !== null);
}

async function processFile(filePath, existingEmbeddings) {
  const content = await fs.promises.readFile(filePath, 'utf8');
  const chunkedDoc = JSON.parse(content);
  const { filename, metadata } = chunkedDoc;

  console.log(`Processing file: ${filename}, Metadata:`, metadata);
  
  const embeddings = [];
  for (let i = 0; i < chunkedDoc.chunks.length; i += BATCH_SIZE) {
    const batch = chunkedDoc.chunks.slice(i, i + BATCH_SIZE);
    const batchResults = await processChunkBatch(batch, filename, metadata, existingEmbeddings);
    embeddings.push(...batchResults);
    await delay(DELAY_MS);
  }

  return embeddings;
}

async function saveCheckpoint(embeddings, outputPath) {
  await fs.promises.writeFile(outputPath, JSON.stringify(embeddings, null, 2));
  console.log(`Checkpoint saved to ${outputPath}`);
}

async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.resolve('data/processed/chunked_documents');
    const outputPath = path.resolve('data/embeddings/aviation_embeddings.json');
    const checkpointPath = path.resolve('data/embeddings/checkpoint.json');
    const files = await fs.promises.readdir(chunkedDocsPath);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    if (jsonFiles.length === 0) {
      console.error('No JSON files found in the chunked_documents directory.');
      return;
    }

    let allEmbeddings = loadExistingEmbeddings(outputPath);
    const processedFiles = new Set(allEmbeddings.map(e => e.filename));

    console.log(`Found ${jsonFiles.length} files to process.`);
    for (const file of jsonFiles) {
      if (processedFiles.has(file)) {
        console.log(`Skipping already processed file: ${file}`);
        continue;
      }

      const filePath = path.join(chunkedDocsPath, file);
      const embeddings = await processFile(filePath, allEmbeddings);
      allEmbeddings = allEmbeddings.concat(embeddings);
      
      // Save checkpoint after each file
      await saveCheckpoint(allEmbeddings, checkpointPath);
    }

    await saveCheckpoint(allEmbeddings, outputPath);
    console.log(`All embeddings saved to ${outputPath}`);

    // Generate CSV report
    await generateCSVReport(allEmbeddings);

  } catch (err) {
    console.error('Error while generating embeddings:', err);
  }
}

async function generateCSVReport(embeddings) {
  const csvWriter = createObjectCsvWriter({
    path: 'data/embeddings/embeddings_report.csv',
    header: [
      {id: 'chunk_id', title: 'Chunk ID'},
      {id: 'filename', title: 'Filename'},
      {id: 'tokens', title: 'Tokens'},
      {id: 'text', title: 'Text'}
    ]
  });

  const records = embeddings.map(e => ({
    chunk_id: e.chunk_id,
    filename: e.filename,
    tokens: e.tokens,
    text: e.text.substring(0, 100) + '...' // Truncate text for readability
  }));

  await csvWriter.writeRecords(records);
  console.log('CSV report generated: data/embeddings/embeddings_report.csv');
}

generateEmbeddings();