import fs from 'fs';
import path from 'path';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { createObjectCsvWriter } from 'csv-writer';
import { fileURLToPath } from 'url';

dotenv.config();

// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Navigate to the project root directory
const projectRoot = path.resolve(__dirname, '..', '..', '..');

// Set up paths
// Define paths relative to the project root
const chunkedDocsPath = path.join(projectRoot, 'data', 'processed', 'chunked_documents');
const outputPath = path.join(projectRoot, 'data', 'embeddings', 'aviation_embeddings.json');
const checkpointPath = path.join(projectRoot, 'data', 'embeddings', 'checkpoint.json');

// Initialize OpenAI API
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Load existing embeddings if available
let existingEmbeddings = {};
if (fs.existsSync(outputPath)) {
    try {
        const fileContent = fs.readFileSync(outputPath, "utf-8");
        existingEmbeddings = JSON.parse(fileContent);
    } catch (error) {
        console.error("Error loading existing embeddings:", error);
        existingEmbeddings = {};
    }
}

// Function to check if a chunk ID already has an embedding
async function isChunkIdExists(chunkId) {
    return existingEmbeddings.hasOwnProperty(chunkId);
}

// Function to generate embeddings for a single chunk
async function generateEmbedding(text) {
    try {
        const response = await openai.embeddings.create({
            model: "text-embedding-ada-002",
            input: text
        });
        return response.data[0].embedding;
    } catch (error) {
        console.error("Error generating embedding:", error);
        return null;
    }
}

// Function to process a single chunk
async function processChunk(chunk, filename, chunkId, metadata) {
    if (isChunkIdExists(chunkId)) {
        console.log(`Skipping duplicate chunk ID: ${chunkId}`);
        return null;
    }

    const embedding = await generateEmbedding(chunk.text);
    if (!embedding) return null;

    return {
        chunk_id: chunkId,
        filename,
        metadata,
        text: chunk.text,
        tokens: chunk.tokens,
        embedding: embedding,
    };
}

// Function to process all chunks in batch
async function processFile(filePath) {
    try {
        const rawData = fs.readFileSync(filePath, "utf-8");
        const document = JSON.parse(rawData);

        console.log(`Processing file: ${document.filename}`);
        const embeddings = [];

        for (const chunk of document.chunks) {
            const chunkId = chunk.chunk_id;
            const metadata = document.metadata || {};
            const processedChunk = await processChunk(chunk, document.filename, chunkId, metadata);
            if (processedChunk) embeddings.push(processedChunk);
        }

        return embeddings.filter(emb => emb !== null);
    } catch (error) {
        console.error(`Error processing file ${filePath}:`, error);
        return [];
    }
}

async function saveCheckpoint(embeddings, outputPath) {
  await fs.promises.writeFile(outputPath, JSON.stringify(embeddings, null, 2));
  console.log(`Checkpoint saved to ${outputPath}`);
}

// Main function to process all chunked documents
async function generateEmbeddings() {
    // Log the paths for debugging
    console.log('Project Root:', projectRoot);
    console.log('Chunked Docs Path:', chunkedDocsPath);
    const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith(".json"));
    let newEmbeddings = [];

    for (const file of files) {
        const filePath = path.join(chunkedDocsPath, file);
        const embeddings = await processFile(filePath);
        newEmbeddings = newEmbeddings.concat(embeddings);
    }

    // Save new embeddings to file
    if (newEmbeddings.length > 0) {
        console.log(` Generated ${newEmbeddings.length} new embeddings.`);

        // Define updatedEmbeddings correctly
        let updatedEmbeddings = { ...existingEmbeddings }; 

        newEmbeddings.forEach(emb => updatedEmbeddings[emb.chunk_id] = emb);

        fs.writeFileSync(outputPath, JSON.stringify(updatedEmbeddings, null, 2), "utf-8");
        console.log(" Updated embeddings file.");

        // Generate CSV report for new embeddings
        await generateCSVReport(newEmbeddings);

        // Save checkpoint after each file
        await saveCheckpoint(updatedEmbeddings, checkpointPath);

    } else {
        console.log(" No new embeddings were generated.");
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
