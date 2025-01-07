const express = require("express");
const bodyParser = require("body-parser");
const { Client } = require("cassandra-driver");
const { Configuration, OpenAIApi } = require("openai");

require("dotenv").config();

const app = express();
app.use(bodyParser.json());

const client = new Client({
  cloud: { secureConnectBundle: "./config/secure-connect-database.zip" },
  credentials: {
    username: process.env.ASTRA_DB_CLIENT_ID,
    password: process.env.ASTRA_DB_CLIENT_SECRET,
  },
});

const openai = new OpenAIApi(
  new Configuration({
    apiKey: process.env.OPENAI_API_KEY,
  })
);

client.connect().then(() => console.log("Connected to AstraDB"));

/**
 * Utility function to compute cosine similarity between two vectors.
 */
function cosineSimilarity(vectorA, vectorB) {
  const dotProduct = vectorA.reduce((sum, a, idx) => sum + a * vectorB[idx], 0);
  const magnitudeA = Math.sqrt(vectorA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vectorB.reduce((sum, b) => sum + b * b, 0));
  return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
}

app.post("/query", async (req, res) => {
  const { userQuery } = req.body;

  if (!userQuery) {
    return res.status(400).send("Query text is required.");
  }

  try {
    // Step 1: Generate embedding for the query
    const queryEmbeddingResponse = await openai.createEmbedding({
      model: "text-embedding-ada-002",
      input: userQuery,
    });
    const queryEmbedding = queryEmbeddingResponse.data.data[0].embedding;

    // Step 2: Fetch all documents and embeddings from AstraDB
    const query = "SELECT chunk_id, filename, text, embedding FROM aviation_data.aviation_documents";
    const result = await client.execute(query);
    const documents = result.rows.map((row) => ({
      chunk_id: row.chunk_id,
      filename: row.filename,
      text: row.text,
      embedding: row.embedding.toJSON().data, // Convert blob to array
    }));

    // Step 3: Calculate similarity
    const similarities = documents.map((doc) => ({
      ...doc,
      similarity: cosineSimilarity(queryEmbedding, doc.embedding),
    }));

    // Step 4: Sort by similarity and return top results
    const sortedResults = similarities.sort((a, b) => b.similarity - a.similarity).slice(0, 10);

    res.status(200).json({ results: sortedResults });
  } catch (error) {
    console.error("Error querying documents:", error);
    res.status(500).send("Error querying documents.");
  }
});

// Utility: Basic health check endpoint
app.get("/", (req, res) => {
  res.send("Server is running!");
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
