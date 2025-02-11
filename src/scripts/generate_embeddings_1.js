import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { createObjectCsvWriter } from 'csv-writer';
import winston from 'winston';
import { format } from 'date-fns';
import { fileURLToPath } from 'url';

// Define __dirname manually for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();

// Set up logging
const logDir = path.resolve(__dirname, '../../logs');
if (!fs.existsSync(logDir)) {
  fs.mkdirSync(logDir, { recursive: true });
}

const logFileName = `generate_embeddings_${format(new Date(), 'yyyy-MM-dd')}.log`;
const logFilePath = path.join(logDir, logFileName);

console.log(`Logging to: ${logFilePath}`);  // Add this for debugging


const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message }) => {
      return `${timestamp} [${level}]: ${message}`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ 
      filename: logFilePath,
      maxsize: 5242880, // 5MB
      maxFiles: 5,
      tailable: true
    })
  ]
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,

});

const BATCH_SIZE = 10; // Adjust based on API limits and performance
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

function isChunkIdExists(chunkId, existingEmbeddings) {
    logger.info(`Checking chunk ID existence: ${chunkId}`);
    return existingEmbeddings.some(e => e.chunk_id === chunkId);
  }

  async function processChunk(chunk, filename, chunkId, metadata, existingEmbeddings) {
    if (isChunkIdExists(chunkId, existingEmbeddings)) {
      logger.info(`Skipping duplicate chunk ID: ${chunkId}`);
      return null;
    }
  
    let attempt = 1;
    while (attempt <= MAX_RETRIES) {
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
        logger.error(`Error generating embedding for chunk ID: ${chunkId} (Attempt ${attempt})`, err);
  
        if (err.response?.status === 429) {
          logger.warn(`Rate limit hit, waiting ${DELAY_MS * attempt}ms...`);
        }
  
        if (attempt === MAX_RETRIES) {
          logger.error(`Failed after ${MAX_RETRIES} attempts.`);
          return null;
        }
  
        await delay(DELAY_MS * attempt);
        attempt++;
      }
    }
  }

  async function processChunkBatch(chunks, filename, metadata) {
    const promises = chunks.map(chunk =>
      processChunk(chunk, filename, chunk.chunk_id, metadata)
    );
  
    const results = await Promise.allSettled(promises);
    return results
      .filter(r => r.status === "fulfilled" && r.value !== null)
      .map(r => r.value);
  }
 

async function processFile(filePath, existingEmbeddings) {
  const content = await fs.promises.readFile(filePath, 'utf8');
  const chunkedDoc = JSON.parse(content);
  const { filename, metadata } = chunkedDoc;

  logger.info(`Processing file: ${filename}, Metadata:`, metadata);
  
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
    try {
      const tempPath = `${outputPath}.tmp`;
      await fs.promises.writeFile(tempPath, JSON.stringify(embeddings, null, 2));
      await fs.promises.rename(tempPath, outputPath);
      logger.info(`Checkpoint saved successfully to ${outputPath}`);
    } catch (err) {
      logger.error(`Error saving checkpoint: ${err}`);
    }
  }
 

async function generateEmbeddings() {
  try {
    const chunkedDocsPath = path.resolve('data/processed/chunked_documents');
    const outputPath = path.resolve('data/embeddings/aviation_embeddings.json');
    const checkpointPath = path.resolve('data/embeddings/checkpoint.json');
    const files = await fs.promises.readdir(chunkedDocsPath);
    const jsonFiles = files.filter(file => file.endsWith('.json'));

    if (jsonFiles.length === 0) {
      logger.error('No JSON files found in the chunked_documents directory.');
      return;
    }

    let allEmbeddings = loadExistingEmbeddings(outputPath);
    const processedFiles = new Set(allEmbeddings.map(e => e.filename));

    logger.info(`Found ${jsonFiles.length} files to process.`);
    for (const file of jsonFiles) {
      if (processedFiles.has(file)) {
        logger.info(`Skipping already processed file: ${file}`);
        continue;
      }

      const filePath = path.join(chunkedDocsPath, file);
      const embeddings = await processFile(filePath, allEmbeddings);
      allEmbeddings = allEmbeddings.concat(embeddings);
      
      // Save checkpoint after each file
      await saveCheckpoint(allEmbeddings, checkpointPath);
    }

    await saveCheckpoint(allEmbeddings, outputPath);
    logger.info(`All embeddings saved to ${outputPath}`);

    // Generate CSV report
    await generateCSVReport(allEmbeddings);

  } catch (err) {
    logger.error('Error while generating embeddings:', err);
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
  logger.info('CSV report generated: data/embeddings/embeddings_report.csv');
  logger.info('Embeddings Generation Complete');
}

generateEmbeddings();