import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

// Get the directory of the current module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..', '..');
const EMBEDDINGS_FILE = path.join(projectRoot, 'data', 'embeddings', 'aviation_embeddings.json');
const CHUNKS_DIR = path.join(projectRoot, 'data', 'processed', 'chunked_documents');

function loadExistingEmbeddings() {
  try {
    const data = fs.readFileSync(EMBEDDINGS_FILE, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.log('No existing embeddings file found. Will generate all embeddings.');
    
    return [];
  }
}

function scanChunkedDocuments() {
  const chunkFiles = fs.readdirSync(CHUNKS_DIR).filter(file => file.endsWith('_chunks.json'));
  return chunkFiles.map(file => {
    const data = fs.readFileSync(path.join(CHUNKS_DIR, file), 'utf8');
    return JSON.parse(data);
  });
}

function findNewChunks(existingEmbeddings, chunkedDocuments) {
  const existingChunks = new Set(existingEmbeddings.map(e => `${e.filename}_${e.chunk_id}`));
  const newChunks = [];

  chunkedDocuments.forEach(doc => {
    doc.chunks.forEach(chunk => {
      const chunkKey = `${doc.filename}_${chunk.chunk_id}`;
      if (!existingChunks.has(chunkKey)) {
        newChunks.push({ filename: doc.filename, chunk_id: chunk.chunk_id });
      }
    });
  });
  console.log(`Found ${newChunks.length} new chunk(s) to process.`);
  return newChunks;
}

function generateEmbeddings() {
  console.log('Generating embeddings for new chunks...');
  exec('node src/scripts/js_files/generate_embeddings.js', (error, stdout, stderr) => {
    if (error) {
      console.error(`Error: ${error.message}`);
      return;
    }
    if (stderr) {
      console.error(`Stderr: ${stderr}`);
      return;
    }
    console.log(`Stdout: ${stdout}`);
  });
}

function main() {
  const existingEmbeddings = loadExistingEmbeddings();
  const chunkedDocuments = scanChunkedDocuments();
  const newChunks = findNewChunks(existingEmbeddings, chunkedDocuments);

  if (newChunks.length > 0) {
    console.log(`Found ${newChunks.length} new chunks. Generating embeddings...`);
    generateEmbeddings();
  } else {
    console.log('No new chunks found. Skipping embedding generation.');
  }
  
}

main();