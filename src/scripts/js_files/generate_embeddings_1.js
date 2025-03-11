import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { OpenAI } from 'openai';
import dotenv from 'dotenv';
import { createObjectCsvWriter } from 'csv-writer';

// Configuration setup
dotenv.config();
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// OpenAI configuration
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// File paths
const dataDir = path.join(__dirname, '../../../data');
const processedDir = path.join(dataDir, 'processed');
const embeddingsDir = path.join(dataDir, 'embeddings');
const checkpointPath = path.join(embeddingsDir, 'checkpoint.json');
const checkpointBackupPath = path.join(embeddingsDir, 'checkpoint.backup.json');
const checksumPath = path.join(embeddingsDir, 'checkpoint.checksum');
const outputPath = path.join(embeddingsDir, 'aviation_embeddings.json');

// Processing constants
const BATCH_SIZE = 5;
const DELAY_MS = 1000;
const MAX_RETRIES = 3;

// Utility functions for robust file operations
async function writeFileAtomically(filePath, data) {
  const tempPath = `${filePath}.tmp`;
  await fs.writeFile(tempPath, data);
  await fs.rename(tempPath, filePath);
}

async function calculateChecksum(data) {
  return crypto.createHash('sha256').update(data).digest('hex');
}

async function saveChecksum(data, checksumPath) {
  const checksum = await calculateChecksum(data);
  await writeFileAtomically(checksumPath, checksum);
  return checksum;
}

async function verifyChecksum(filePath, checksumPath) {
  try {
    const data = await fs.readFile(filePath, 'utf8');
    const storedChecksum = await fs.readFile(checksumPath, 'utf8');
    const calculatedChecksum = await calculateChecksum(data);
    return calculatedChecksum === storedChecksum;
  } catch (error) {
    console.warn(`Checksum verification failed: ${error.message}`);
    return false;
  }
}

async function backupCheckpoint() {
  try {
    const exists = await fs.access(checkpointPath)
      .then(() => true)
      .catch(() => false);
    
    if (exists) {
      console.log('Creating backup of existing checkpoint...');
      const data = await fs.readFile(checkpointPath, 'utf8');
      await writeFileAtomically(checkpointBackupPath, data);
      console.log('Backup created successfully.');
    }
  } catch (error) {
    console.error(`Failed to backup checkpoint: ${error.message}`);
  }
}

async function saveCheckpoint(processedFiles) {
  try {
    console.log('Saving checkpoint...');
    const checkpointData = JSON.stringify(processedFiles, null, 2);
    
    // Create backup of current checkpoint if it exists
    await backupCheckpoint();
    
    // Save new checkpoint atomically
    await writeFileAtomically(checkpointPath, checkpointData);
    
    // Save checksum for verification
    await saveChecksum(checkpointData, checksumPath);
    
    console.log('Checkpoint saved successfully.');
  } catch (error) {
    console.error(`Failed to save checkpoint: ${error.message}`);
    throw error;
  }
}

async function loadCheckpoint() {
  try {
    const checkpointExists = await fs.access(checkpointPath)
      .then(() => true)
      .catch(() => false);
    
    if (!checkpointExists) {
      console.log('No checkpoint found, starting fresh.');
      return [];
    }
    
    // Verify checksum
    const checksumValid = await verifyChecksum(checkpointPath, checksumPath);
    
    if (!checksumValid) {
      console.warn('Checkpoint checksum validation failed!');
      
      // Try to use backup
      const backupExists = await fs.access(checkpointBackupPath)
        .then(() => true)
        .catch(() => false);
      
      if (backupExists) {
        console.log('Attempting to restore from backup...');
        const backupData = await fs.readFile(checkpointBackupPath, 'utf8');
        await writeFileAtomically(checkpointPath, backupData);
        await saveChecksum(backupData, checksumPath);
        console.log('Restored from backup successfully.');
        return JSON.parse(backupData);
      } else {
        console.error('No valid backup found, starting fresh.');
        return [];
      }
    }
    
    console.log('Loading checkpoint...');
    const checkpointData = await fs.readFile(checkpointPath, 'utf8');
    return JSON.parse(checkpointData);
  } catch (error) {
    console.error(`Failed to load checkpoint: ${error.message}`);
    return [];
  }
}

async function getOpenAIEmbedding(text, retries = 0) {
  try {
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: text,
    });
    
    return embeddingResponse.data[0].embedding;
  } catch (error) {
    if (retries < MAX_RETRIES) {
      console.warn(`OpenAI API error, retrying (${retries + 1}/${MAX_RETRIES}): ${error.message}`);
      await delay(DELAY_MS * (retries + 1)); // Exponential backoff
      return getOpenAIEmbedding(text, retries + 1);
    }
    throw error;
  }
}

// Add delay function for rate limiting and retries
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Process a single chunk of text
async function processChunk(chunk, filename, chunkId, existingEmbeddings = []) {
  try {
    // Check if we already have this chunk processed
    const existingChunk = existingEmbeddings.find(
      item => item.filename === filename && item.chunk_id === chunkId
    );
    
    if (existingChunk) {
      console.log(`Chunk ${chunkId} from ${filename} already exists, skipping`);
      return existingChunk;
    }
    
    console.log(`Processing chunk ${chunkId} from ${filename}`);
    
    // Get embedding for the chunk
    const embeddingVector = await getOpenAIEmbedding(chunk.text);
    
    // Return the chunk with its embedding in the format compatible with AstraDB
    return {
      chunk_id: chunkId,
      filename: filename,
      text: chunk.text,
      tokens: chunk.tokens || chunk.text.split(/\s+/).length, // Estimate tokens if not provided
      embedding: embeddingVector
    };
  } catch (error) {
    console.error(`Failed to process chunk ${chunkId} from ${filename}: ${error.message}`);
    throw error;
  }
}

// Process a batch of chunks in parallel
async function processChunkBatch(chunks, filename, existingEmbeddings = []) {
  try {
    console.log(`Processing batch of ${chunks.length} chunks from ${filename}`);
    
    // Process chunks with a small delay between each to avoid rate limits
    const results = [];
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      const result = await processChunk(chunk, filename, chunk.id || `${filename}-${i}`, existingEmbeddings);
      results.push(result);
      
      // Add delay between chunks except for the last one
      if (i < chunks.length - 1) {
        await delay(DELAY_MS);
      }
    }
    
    return results;
  } catch (error) {
    console.error(`Failed to process chunk batch from ${filename}: ${error.message}`);
    throw error;
  }
}

async function ensureDirectoriesExist() {
  await fs.mkdir(embeddingsDir, { recursive: true });
}

async function getFilesToProcess() {
  try {
    const files = await fs.readdir(processedDir);
    return files.filter(file => file.endsWith('.json'));
  } catch (error) {
    console.error(`Failed to read directory: ${error.message}`);
    return [];
  }
}

async function processFile(file, processedFiles, existingEmbeddings = []) {
  // Skip files already processed
  if (processedFiles.includes(file)) {
    console.log(`Skipping already processed file: ${file}`);
    return null;
  }

  try {
    console.log(`Processing file: ${file}`);
    const filePath = path.join(processedDir, file);
    const fileContent = await fs.readFile(filePath, 'utf8');
    const jsonData = JSON.parse(fileContent);
    
    const fileId = jsonData.id || file.replace('.json', '');
    const content = jsonData.content || '';
    
    // Check if content has chunks
    if (jsonData.chunks && Array.isArray(jsonData.chunks) && jsonData.chunks.length > 0) {
      console.log(`Processing ${jsonData.chunks.length} chunks from ${file}`);
      
      // Process chunks in batches
      const allEmbeddings = [];
      for (let i = 0; i < jsonData.chunks.length; i += BATCH_SIZE) {
        const batchChunks = jsonData.chunks.slice(i, i + BATCH_SIZE);
        const batchResults = await processChunkBatch(
          batchChunks, 
          fileId, 
          existingEmbeddings
        );
        allEmbeddings.push(...batchResults);
        
        // Add delay between batches
        if (i + BATCH_SIZE < jsonData.chunks.length) {
          await delay(DELAY_MS);
        }
      }
      
      return allEmbeddings;
    } else {
      // If no chunks, process the whole content as a single embedding
      console.log(`Processing ${file} as a single document (no chunks found)`);
      
      // Create a pseudo-chunk for the whole document
      const singleChunk = {
        id: 'single',
        text: content,
        tokens: content.split(/\s+/).length // Estimate token count
      };
      
      const result = await processChunk(
        singleChunk,
        fileId,
        `${fileId}-single`,
        existingEmbeddings
      );
      
      return [result];
    }
  } catch (error) {
    console.error(`Failed to process file ${file}: ${error.message}`);
    return null;
  }
}

async function generateEmbeddings() {
  try {
    console.log('Starting embedding generation process...');
    await ensureDirectoriesExist();
    
    // Load checkpoint
    let processedFiles = await loadCheckpoint();
    console.log(`Loaded ${processedFiles.length} processed files from checkpoint.`);
    
    // Get files to process
    const files = await getFilesToProcess();
    console.log(`Found ${files.length} files to process.`);
    
    // Process files and generate embeddings
    const embeddings = [];
    
    // Load existing embeddings if available
    let existingEmbeddings = [];
    try {
      const outputExists = await fs.access(outputPath)
        .then(() => true)
        .catch(() => false);
      
      if (outputExists) {
        const existingData = await fs.readFile(outputPath, 'utf8');
        existingEmbeddings = JSON.parse(existingData);
        console.log(`Loaded ${existingEmbeddings.length} existing embeddings.`);
      }
    } catch (error) {
      console.warn(`Failed to load existing embeddings: ${error.message}`);
    }
    
    for (const file of files) {
      const embeddingResults = await processFile(file, processedFiles, existingEmbeddings);
      
      if (embeddingResults) {
        // If the result is an array (chunked processing), extend the embeddings array
        if (Array.isArray(embeddingResults)) {
          embeddings.push(...embeddingResults);
        } else {
          // If not an array (backward compatibility), add as a single item
          embeddings.push(embeddingResults);
        }
        
        processedFiles.push(file);
        
        // Save checkpoint after each file to ensure progress is not lost
        await saveCheckpoint(processedFiles);
      }
    }
    
    // We already loaded existing embeddings earlier in the process
    
    // Merge existing embeddings with new ones
    const allEmbeddings = [...existingEmbeddings, ...embeddings];
    
    // Remove duplicates by ID
    const uniqueEmbeddings = [];
    const seenIds = new Set();
    
    for (const embedding of allEmbeddings) {
      if (!seenIds.has(embedding.id)) {
        seenIds.add(embedding.id);
        uniqueEmbeddings.push(embedding);
      }
    }
    
    // Save final embeddings atomically
    const outputData = JSON.stringify(uniqueEmbeddings, null, 2);
    await writeFileAtomically(outputPath, outputData);
    
    // Save checksum for final output
    await saveChecksum(outputData, `${outputPath}.checksum`);
    
    console.log(`Successfully generated embeddings for ${embeddings.length} files.`);
    console.log(`Total unique embeddings: ${uniqueEmbeddings.length}`);
    

    // Generate CSV report
    await generateCSVReport(allEmbeddings);
    
    return uniqueEmbeddings;
  } catch (error) {
    console.error(`Embedding generation failed: ${error.message}`);
    throw error;
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

// Execute the main function
generateEmbeddings()
  .then(() => {
    console.log('Embedding generation completed successfully.');
  })
  .catch((error) => {
    console.error(`Embedding generation failed: ${error.stack}`);
    process.exit(1);
  });

await fs.writeFile(outputPath, JSON.stringify(allEmbeddings, null, 2));

// Export for potential reuse in other modules
export { 
  generateEmbeddings, 
  getOpenAIEmbedding, 
  processChunk, 
  processChunkBatch, 
  processFile 
};

