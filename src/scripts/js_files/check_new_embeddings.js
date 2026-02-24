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
const ASTRA_CONTENT_FILE = path.join(projectRoot, "data", "astra_db", "astra_db_content.json");
const STORE_SCRIPT = path.join(projectRoot, "src", "scripts", "js_files", "store_embeddings_astra.js");

function loadEmbeddings(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      console.log(`File ${filePath} does not exist. Assuming empty dataset.`);
      return [];
    }
    const data = fs.readFileSync(filePath, "utf8");
    return JSON.parse(data);
  } catch (error) {
    console.log(`Error reading file ${filePath}: ${error.message}`);
    return [];
  }
}

function findNewEmbeddings(localEmbeddings, astraEmbeddings) {
  const astraChunkIds = new Set(astraEmbeddings.map((embedding) => embedding.chunk_id));
  return localEmbeddings.filter((embedding) => !astraChunkIds.has(embedding.chunk_id));
}

async function storeEmbeddings() {
  console.log("Storing new embeddings in Astra DB...");
  const { stdout, stderr } = await execFileAsync("node", [STORE_SCRIPT], {
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
  const localEmbeddings = loadEmbeddings(EMBEDDINGS_FILE);
  const astraEmbeddings = loadEmbeddings(ASTRA_CONTENT_FILE);

  const newEmbeddings = findNewEmbeddings(localEmbeddings, astraEmbeddings);
  if (newEmbeddings.length > 0) {
    console.log(`Found ${newEmbeddings.length} new embeddings. Storing in Astra DB...`);
    await storeEmbeddings();
  } else {
    console.log("No new embeddings found. Skipping Astra DB update.");
  }
}

main().catch((error) => {
  console.error("Error in check_new_embeddings:", error);
  process.exit(1);
});
