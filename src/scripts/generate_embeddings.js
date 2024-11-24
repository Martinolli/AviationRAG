const fs = require('fs');
const path = require('path');
const dotenv = require('dotenv');
const { Configuration, OpenAIApi } = require('openai');

// Log to check the imported objects
console.log('Configuration:', Configuration);
console.log('OpenAIApi:', OpenAIApi);

// Load environment variables from .env
dotenv.config();

// Initialize OpenAI client using the OpenAI SDK
const configuration = new Configuration({
apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

async function generateEmbeddings() {
  try {
    // Load chunked documents
    const dataPath = path.join(__dirname, '../../data/processed/chunked_aviation_corpus.json');
    const rawData = fs.readFileSync(dataPath, 'utf-8');
    const chunkedDocs = JSON.parse(rawData);

    const embeddings = [];

    // Function to add delay between API calls
    function delay(ms) {
      return new Promise(resolve => setTimeout(resolve, ms));
    }

    // Loop over each chunk to generate embeddings
    for (const chunk of chunkedDocs) {
      try {
        const response = await openai.createEmbedding({
          model: 'text-embedding-ada-002', // The embedding model to use
          input: chunk.text_chunk,
        });

        const embeddingVector = response.data.data[0].embedding; // Extract the embedding from the response

        embeddings.push({
          id: chunk.id,
          title: chunk.title,
          text_chunk: chunk.text_chunk,
          embedding: embeddingVector,
        });

        console.log(`Generated embedding for chunk ID: ${chunk.id}`);

        // Add delay to avoid hitting rate limits
        await delay(1000); // Adjust delay as needed
      } catch (err) {
        console.error(`Error generating embedding for chunk ID: ${chunk.id}`, err);
      }
    }

    // Save embeddings to a new JSON file for later processing
    const outputPath = path.join(__dirname, '../../data/processed/aviation_embeddings.json');
    const fsPromises = fs.promises;
    await fsPromises.writeFile(outputPath, JSON.stringify(embeddings, null, 2));
    console.log(`Embeddings saved to ${outputPath}`);

  } catch (err) {
    console.error('Error while generating embeddings:', err);
  }
}

// Run the function
generateEmbeddings();
