import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { exec } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..', '..', '..');

const EMBEDDINGS_FILE = path.join(projectRoot, 'data', 'embeddings', 'aviation_embeddings.json');
const ASTRA_CONTENT_FILE = path.join(projectRoot, 'data', 'astra_db', 'astra_db_content.json');

function loadEmbeddings(filePath) {
  try {
    if (!fs.existsSync(filePath)) {
      console.log(`File ${filePath} does not exist. Assuming empty database.`);
      return [];
    }
    const data = fs.readFileSync(filePath, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.log(`Error reading file ${filePath}: ${error.message}`);
    return [];
  }
}

function findNewEmbeddings(localEmbeddings, astraEmbeddings) {
  const astraChunkIds = new Set(astraEmbeddings.map(e => e.chunk_id));
  return localEmbeddings.filter(embedding => !astraChunkIds.has(embedding.chunk_id));
}

function storeEmbeddings() {
  console.log('Storing new embeddings in Astra DB...');
  exec('node src/scripts/js_files/store_embeddings_astra.js', (error, stdout, stderr) => {
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
  const localEmbeddings = loadEmbeddings(EMBEDDINGS_FILE);
  const astraEmbeddings = loadEmbeddings(ASTRA_CONTENT_FILE);

  const newEmbeddings = findNewEmbeddings(localEmbeddings, astraEmbeddings);

  if (newEmbeddings.length > 0) {
    console.log(`Found ${newEmbeddings.length} new embeddings. Storing in Astra DB...`);
    storeEmbeddings();
  } else {
    console.log('No new embeddings found. Skipping Astra DB update.');
  }
}

main();