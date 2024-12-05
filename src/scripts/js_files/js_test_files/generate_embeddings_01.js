const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');

// Load environment variables from .env
dotenv.config();

// Initialize OpenAI client using the OpenAI SDK
const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

// Log to check the initialized OpenAI client
console.log('OpenAI client initialized:', openai);

async function generateEmbeddings() {
  try {
    // Path to the folder containing chunked JSON files
    const chunkedDocsPath = path.join(__dirname, '../../data/processed/chunked_documents');
    const outputPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');

    const embeddings = [];

    // Function to add delay between API calls
    function delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Read all JSON files in the chunked documents folder
    const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith('.json'));

    for (const file of files) {
      const filePath = path.join(chunkedDocsPath, file);

      // Read and parse the JSON file
      const rawData = fs.readFileSync(filePath, 'utf-8');
      const chunkedDocs = JSON.parse(rawData);

      console.log(`Processing file: ${file}`);

      // Process each chunk in the file
      for (const chunk of chunkedDocs.chunks) {
        try {
          const response = await openai.createEmbedding({
            model: 'text-embedding-ada-002', // The embedding model to use
            input: chunk.text,
          });

          const embeddingVector = response.data.data[0].embedding; // Extract the embedding from the response

          embeddings.push({
            chunk_id: chunk.chunk_id,
            filename: chunkedDocs.filename,
            text: chunk.text,
            embedding: embeddingVector,
          });

          console.log(`Generated embedding for chunk ID: ${chunk.chunk_id}`);

          // Add delay to avoid hitting rate limits
          await delay(1000); // Adjust delay as needed
        } catch (err) {
          console.error(`Error generating embedding for chunk ID: ${chunk.chunk_id}`, err);
        }
      }
    }

    // Save embeddings to a new JSON file for later processing
    await fs.promises.writeFile(outputPath, JSON.stringify(embeddings, null, 2));
    console.log(`Embeddings saved to ${outputPath}`);

  } catch (err) {
    console.error('Error while generating embeddings:', err);
  }
}

// Run the function
generateEmbeddings();
