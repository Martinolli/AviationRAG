const fs = require('fs');
const path = require('path');

// Parameters for chunk size
const CHUNK_SIZE = 512; // Adjust this value based on your tokenization needs

async function chunkDocuments() {
  try {
    // Load raw aviation corpus JSON file
    const dataPath = path.join(__dirname, '../../data/processed/aviation_corpus.json'); // Adjust if needed
    const rawData = fs.readFileSync(dataPath, 'utf-8');
    const aviationCorpus = JSON.parse(rawData);

    const chunks = [];

    // Loop over each document
    aviationCorpus.forEach((doc, index) => {
      const text = doc.text; // Assuming each document has a "text" field

      // Break text into chunks of size CHUNK_SIZE
      for (let i = 0; i < text.length; i += CHUNK_SIZE) {
        const chunkText = text.slice(i, i + CHUNK_SIZE);
        chunks.push({
          id: `${index}-${i}`, // Unique identifier
          title: doc.title || `Doc_${index}`, // Optional title
          text_chunk: chunkText,
        });
      }
    });

    // Save chunks to a new JSON file
    const outputPath = path.join(__dirname, '../../data/processed/chunked_aviation_corpus.json');
    fs.writeFileSync(outputPath, JSON.stringify(chunks, null, 2));
    console.log(`Chunks saved to ${outputPath}`);
  } catch (err) {
    console.error('Error while chunking documents:', err);
  }
}

// Run the chunking function
chunkDocuments();
