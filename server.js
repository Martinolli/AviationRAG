import express from "express";
import bodyParser from "body-parser";
import { Client } from "cassandra-driver";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(bodyParser.json());

const client = new Client({
  cloud: { secureConnectBundle: "./config/secure-connect-aviation-rag-db.zip" },
  credentials: {
    username: process.env.ASTRA_DB_CLIENT_ID,
    password: process.env.ASTRA_DB_CLIENT_SECRET,
  },
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});


client.connect().then(() => console.log("Connected to AstraDB"));

/**
 * Utility function to compute cosine similarity between two vectors.
 */
function cosineSimilarity(vectorA, vectorB) {
  if (vectorA.length !== vectorB.length) return 0; // Handle dimension mismatch
  const dotProduct = vectorA.reduce((sum, a, idx) => sum + a * vectorB[idx], 0);
  const magnitudeA = Math.sqrt(vectorA.reduce((sum, a) => sum + a * a, 0));
  const magnitudeB = Math.sqrt(vectorB.reduce((sum, b) => sum + b * b, 0));
  return magnitudeA && magnitudeB ? dotProduct / (magnitudeA * magnitudeB) : 0;
}

async function generateChatCompletion(prompt, retries = 3) {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        messages: [{ role: "user", content: prompt }],
      });

      if (!response || !response.choices || response.choices.length === 0) {
        throw new Error("Invalid response structure from OpenAI Chat Completion API");
      }

      return response.choices[0].message.content;
    } catch (error) {
      console.error(`Attempt ${i + 1} failed: ${error.message}`);
      if (i === retries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, 1000)); // Retry after 1 second
    }
  }
}

app.post("/query", async (req, res) => {
  const { userQuery } = req.body;

  if (!userQuery) {
    return res.status(400).json({ error: "Query text is required." });
  }

  try {
    // Step 1: Generate embedding for the query
    console.log("Generating embedding for query:", userQuery);
    const queryEmbeddingResponse = await openai.embeddings.create({
      model: "text-embedding-ada-002",
      input: userQuery,
    });
    console.log("OpenAI Embedding API Response:", JSON.stringify(queryEmbeddingResponse, null, 2));

    if (!queryEmbeddingResponse.data || !queryEmbeddingResponse.data[0]) {
      throw new Error("Invalid response structure from OpenAI Embeddings API");
    }

    const queryEmbedding = queryEmbeddingResponse.data[0].embedding;

    if (!Array.isArray(queryEmbedding) || queryEmbedding.length === 0) {
      throw new Error("Invalid embedding format in OpenAI response");
    }

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

    // Step 4: Sort by similarity and select top results
    const sortedResults = similarities.sort((a, b) => b.similarity - a.similarity).slice(0, 10);

    // Step 5: Generate a consolidated response using OpenAI
    const context = sortedResults.map(doc => doc.text).join("\n\n");
    const prompt = `
    Context:
    ${context}

    Question:
    ${userQuery}

    Provide a detailed, accurate, and concise response based on the context above.
    `;
    const chatResponse = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      messages: [{ role: "user", content: prompt }],
    });
    console.log("OpenAI Chat API Response:", JSON.stringify(chatResponse, null, 2));

    // Validate the response structure
    if (!chatResponse || !chatResponse.choices || chatResponse.choices.length === 0) {
      throw new Error("Invalid response structure from OpenAI Chat Completion API");
    }

    // Extract the generated response
    const generatedResponse = chatResponse.choices[0].message.content;

    // Return the results and generated response
    console.log("Sending response to frontend:", {
      response: generatedResponse,
      results: sortedResults,
    });
    res.status(200).json({
      response: generatedResponse,
      results: sortedResults,
    });
    
  } catch (error) {
    console.error("Error querying documents or generating response:", {
      message: error.message,
      stack: error.stack,
    });
    res.status(500).json({
      error: "Failed to process your query.",
      details: error.message,
    });
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
