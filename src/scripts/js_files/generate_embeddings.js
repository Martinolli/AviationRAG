import fs from "fs";
import path from "path";
import dotenv from "dotenv";
import OpenAI from "openai";
import { createObjectCsvWriter } from "csv-writer";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, "..", "..", "..");

dotenv.config({ path: path.join(projectRoot, ".env") });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const BATCH_SIZE = 5;
const DELAY_MS = 1000;
const MAX_RETRIES = 3;

const chunkedDocsPath = path.join(projectRoot, "data", "processed", "chunked_documents");
const outputPath = path.join(projectRoot, "data", "embeddings", "aviation_embeddings.json");
const checkpointPath = path.join(projectRoot, "data", "embeddings", "checkpoint.json");
const reportPath = path.join(projectRoot, "data", "embeddings", "embeddings_report.csv");

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadJsonFile(filePath, fallbackValue) {
  if (!fs.existsSync(filePath)) {
    return fallbackValue;
  }

  try {
    const rawData = fs.readFileSync(filePath, "utf-8");
    return JSON.parse(rawData);
  } catch (error) {
    console.error(`Failed to parse JSON from ${filePath}:`, error.message);
    return fallbackValue;
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
      await delay(DELAY_MS * (retries + 1));
      return getOpenAIEmbedding(text, retries + 1);
    }
    throw error;
  }
}

async function processChunk(chunk, filename, metadata, existingChunkIds) {
  const chunkId = chunk.chunk_id;
  if (existingChunkIds.has(chunkId)) {
    return null;
  }

  try {
    const embeddingVector = await getOpenAIEmbedding(chunk.text);
    existingChunkIds.add(chunkId);

    console.log(`Generated embedding for chunk ID: ${chunkId}`);
    return {
      chunk_id: chunkId,
      filename,
      metadata,
      text: chunk.text,
      tokens: chunk.tokens,
      embedding: embeddingVector,
    };
  } catch (error) {
    console.error(`Failed generating embedding for chunk ID: ${chunkId}`, error.message);
    return null;
  }
}

async function processChunkBatch(chunks, filename, metadata, existingChunkIds) {
  const batchPromises = chunks.map((chunk) => processChunk(chunk, filename, metadata, existingChunkIds));
  return (await Promise.all(batchPromises)).filter((result) => result !== null);
}

async function processChunkedDocument(chunkedDoc, existingChunkIds) {
  const { filename, metadata } = chunkedDoc;
  const chunks = Array.isArray(chunkedDoc.chunks) ? chunkedDoc.chunks : [];

  console.log(`Processing file: ${filename}`);

  const newEmbeddings = [];
  for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
    const batch = chunks.slice(i, i + BATCH_SIZE);
    const batchResults = await processChunkBatch(batch, filename, metadata, existingChunkIds);
    newEmbeddings.push(...batchResults);
    await delay(DELAY_MS);
  }

  return newEmbeddings;
}

async function saveJsonOutput(data, filePath, message) {
  await fs.promises.mkdir(path.dirname(filePath), { recursive: true });
  await fs.promises.writeFile(filePath, JSON.stringify(data, null, 2), "utf8");
  console.log(`${message}: ${filePath}`);
}

async function generateCSVReport(embeddings, csvPath) {
  const csvWriter = createObjectCsvWriter({
    path: csvPath,
    header: [
      { id: "chunk_id", title: "Chunk ID" },
      { id: "filename", title: "Filename" },
      { id: "tokens", title: "Tokens" },
      { id: "text", title: "Text" },
    ],
  });

  const records = embeddings.map((embedding) => ({
    chunk_id: embedding.chunk_id,
    filename: embedding.filename,
    tokens: embedding.tokens,
    text: `${embedding.text.substring(0, 100)}...`,
  }));

  await csvWriter.writeRecords(records);
  console.log(`CSV report generated: ${csvPath}`);
}

async function generateEmbeddings() {
  try {
    if (!fs.existsSync(chunkedDocsPath)) {
      console.error(`The directory ${chunkedDocsPath} does not exist.`);
      console.log("Please ensure chunk files exist before running embeddings generation.");
      return;
    }

    const files = await fs.promises.readdir(chunkedDocsPath);
    const jsonFiles = files.filter((file) => file.endsWith(".json"));

    if (jsonFiles.length === 0) {
      console.log("No JSON files found in chunked_documents directory.");
      return;
    }

    const allEmbeddings = loadJsonFile(checkpointPath, loadJsonFile(outputPath, []));
    const existingChunkIds = new Set(allEmbeddings.map((embedding) => embedding.chunk_id));

    console.log(`Found ${jsonFiles.length} files to inspect.`);
    for (const file of jsonFiles) {
      const filePath = path.join(chunkedDocsPath, file);
      const chunkedDoc = loadJsonFile(filePath, null);
      if (!chunkedDoc) {
        console.warn(`Skipping unreadable chunk file: ${filePath}`);
        continue;
      }

      const chunks = Array.isArray(chunkedDoc.chunks) ? chunkedDoc.chunks : [];
      const hasNewChunks = chunks.some((chunk) => !existingChunkIds.has(chunk.chunk_id));
      if (!hasNewChunks) {
        console.log(`Skipping fully processed file: ${file}`);
        continue;
      }

      const embeddings = await processChunkedDocument(chunkedDoc, existingChunkIds);
      if (embeddings.length > 0) {
        allEmbeddings.push(...embeddings);
        await saveJsonOutput(allEmbeddings, checkpointPath, "Checkpoint saved");
      }
    }

    await saveJsonOutput(allEmbeddings, outputPath, "Final output saved");

    if (fs.existsSync(checkpointPath)) {
      await fs.promises.unlink(checkpointPath);
      console.log("Checkpoint file removed after successful completion.");
    }

    await generateCSVReport(allEmbeddings, reportPath);
  } catch (error) {
    console.error("Error while generating embeddings:", error);
    if (error.code === "ENOENT") {
      console.log("Please verify project structure and required directories.");
    }
  }
}

generateEmbeddings();
