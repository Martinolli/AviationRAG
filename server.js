// Updated server.js
const { VectorStoreRetriever, OpenAI } = require('langchain');
const { Client } = require('cassandra-driver');
const dotenv = require('dotenv');

dotenv.config();

// Initialize Cassandra Client
const client = new Client({
  cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
  credentials: {
    username: process.env.ASTRA_DB_CLIENT_ID,
    password: process.env.ASTRA_DB_CLIENT_SECRET,
  },
  keyspace: process.env.ASTRA_DB_KEYSPACE,
});

// Initialize OpenAI Client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Define constants
const SIMILARITY_THRESHOLD = 0.8;
const MAX_CHUNKS = 5;
const MAX_TOKENS = 1500;

// Query function
async function handleQuery(userQuery) {
  try {
    // Step 1: Generate Query Embedding
    const queryEmbedding = await generateQueryEmbedding(userQuery);

    // Step 2: Retrieve Relevant Chunks
    const relevantChunks = await retrieveRelevantChunks(queryEmbedding);

    // Step 3: Construct Context
    const context = constructContext(relevantChunks);

    // Step 4: Generate Response from LLM
    const response = await generateLLMResponse(userQuery, context);

    return response;
  } catch (error) {
    console.error("Error in handleQuery:", error);
    return "An error occurred while processing your request.";
  }
}

// Helper function to generate query embedding
async function generateQueryEmbedding(query) {
  const embedding = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: query,
  });
  return embedding.data[0].embedding;
}

// Helper function to retrieve relevant chunks
async function retrieveRelevantChunks(queryEmbedding) {
  const query = `SELECT chunk_id, text, embedding FROM aviation_documents;`;
  const result = await client.execute(query);

  const chunks = result.rows.map(row => ({
    chunk_id: row.chunk_id,
    text: row.text,
    similarity: computeCosineSimilarity(queryEmbedding, row.embedding),
  }));

  return chunks
    .filter(chunk => chunk.similarity >= SIMILARITY_THRESHOLD)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, MAX_CHUNKS);
}

// Helper function to construct context
function constructContext(chunks) {
  return chunks.map(chunk => chunk.text).join("\n\n");
}

// Helper function to generate response
async function generateLLMResponse(query, context) {
  const prompt = `Context:
  ${context}

  Question:
  ${query}

  Provide a detailed and accurate response based on the context.`;

  const response = await openai.chat.create({
    model: "gpt-3.5-turbo",
    prompt: prompt,
    max_tokens: MAX_TOKENS,
    temperature: 0.7,
  });

  return response.choices[0].message.content.trim();
}

// Cosine Similarity Helper
function computeCosineSimilarity(vec1, vec2) {
  return vec1.reduce((sum, val, i) => sum + val * vec2[i], 0) /
         (Math.sqrt(vec1.reduce((sum, val) => sum + val ** 2, 0)) *
          Math.sqrt(vec2.reduce((sum, val) => sum + val ** 2, 0)));
}

module.exports = handleQuery;
