import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { execFile } from "child_process";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..", "..");

const EMBEDDINGS_FILE = path.join(projectRoot, "data", "embeddings", "aviation_embeddings.json");
const CHUNKS_DIR = path.join(projectRoot, "data", "processed", "chunked_documents");
const GENERATE_SCRIPT = path.join(projectRoot, "src", "scripts", "js_files", "generate_embeddings.js");

function loadExistingEmbeddings() {
  try {
    if (!fs.existsSync(EMBEDDINGS_FILE)) {
      console.log("No existing embeddings file found. Will generate all embeddings.");
      return [];
    }
    const data = fs.readFileSync(EMBEDDINGS_FILE, "utf8");
    return JSON.parse(data);
  } catch (error) {
    console.log(`Unable to read embeddings file: ${error.message}`);
    return [];
  }
}

function scanChunkedDocuments() {
  if (!fs.existsSync(CHUNKS_DIR)) {
    console.log(`Chunk directory does not exist: ${CHUNKS_DIR}`);
    return [];
  }

  const chunkFiles = fs.readdirSync(CHUNKS_DIR).filter((file) => file.endsWith("_chunks.json"));
  return chunkFiles
    .map((file) => {
      try {
        const data = fs.readFileSync(path.join(CHUNKS_DIR, file), "utf8");
        return JSON.parse(data);
      } catch (error) {
        console.warn(`Skipping invalid chunk file ${file}: ${error.message}`);
        return null;
      }
    })
    .filter((doc) => doc !== null);
}

function findNewChunks(existingEmbeddings, chunkedDocuments) {
  const existingChunkIds = new Set(existingEmbeddings.map((embedding) => embedding.chunk_id));
  const newChunks = [];

  chunkedDocuments.forEach((doc) => {
    const chunks = Array.isArray(doc.chunks) ? doc.chunks : [];
    chunks.forEach((chunk) => {
      if (!existingChunkIds.has(chunk.chunk_id)) {
        newChunks.push({ filename: doc.filename, chunk_id: chunk.chunk_id });
      }
    });
  });

  console.log(`Found ${newChunks.length} new chunk(s) to process.`);
  return newChunks;
}

async function generateEmbeddings() {
  console.log("Generating embeddings for new chunks...");
  const { stdout, stderr } = await execFileAsync("node", [GENERATE_SCRIPT], {
    cwd: projectRoot,
    maxBuffer: 1024 * 1024 * 20,
  });

  if (stderr) {
    console.error(stderr);
  }
  if (stdout) {
    console.log(stdout);
  }
}

async function main() {
  const existingEmbeddings = loadExistingEmbeddings();
  const chunkedDocuments = scanChunkedDocuments();
  const newChunks = findNewChunks(existingEmbeddings, chunkedDocuments);

  if (newChunks.length > 0) {
    console.log(`Found ${newChunks.length} new chunks. Generating embeddings...`);
    await generateEmbeddings();
  } else {
    console.log("No new chunks found. Skipping embedding generation.");
  }
}

main().catch((error) => {
  console.error("Error in check_new_chunks:", error);
  process.exit(1);
});
