import { NextApiRequest, NextApiResponse } from 'next';
import { ChatOpenAI } from '@langchain/openai';
import { Client } from 'cassandra-driver';
import { OpenAIEmbeddings } from '@langchain/openai';
import { createRetrievalChain } from "langchain/chains/retrieval";
import { Document } from 'langchain/document';
import { BaseRetriever, BaseRetrieverInterface } from "@langchain/core/retrievers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";

class CustomAstraDBRetriever extends BaseRetriever implements BaseRetrieverInterface<Record<string, any>> {
  lc_namespace: string[];
  private client: Client;
  private embeddings: OpenAIEmbeddings;

  constructor(client: Client, embeddings: OpenAIEmbeddings) {
    super();
    this.client = client;
    this.embeddings = embeddings;
  }

  async getRelevantDocuments(query: string): Promise<Document[]> {
    try {
      console.log("Retrieving documents for query:", query);
      const embedding = await this.embeddings.embedQuery(query);
      const queryText = 'SELECT * FROM aviation_documents LIMIT 10';
      const results = await this.client.execute(queryText, [], { prepare: true });

      const documents = results.rows.map((row: any) => new Document({
        pageContent: row.text,
        metadata: { filename: row.filename, chunk_id: row.chunk_id, embedding: row.embedding },
      }));

      const sortedDocuments = documents.sort((a, b) => {
        const simA = this.calculateSimilarity(embedding, a);
        const simB = this.calculateSimilarity(embedding, b);
        return simB - simA;
      });

      console.log("Retrieved and sorted documents:", sortedDocuments);
      return sortedDocuments.slice(0, 5);
    } catch (error) {
      console.error("Error retrieving documents:", error);
      throw new Error("Failed to retrieve documents");
    }
  }

  private calculateSimilarity(queryEmbedding: number[], document: Document): number {
    const docEmbedding = document.metadata.embedding;
    const dotProduct = queryEmbedding.reduce((sum, q, i) => sum + q * docEmbedding[i], 0);
    const queryMagnitude = Math.sqrt(queryEmbedding.reduce((sum, q) => sum + q * q, 0));
    const docMagnitude = Math.sqrt(docEmbedding.reduce((sum, d) => sum + d * d, 0));
    return dotProduct / (queryMagnitude * docMagnitude);
  }

  async invoke(input: string): Promise<Document[]> {
    return this.getRelevantDocuments(input);
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { query } = req.body;
    console.log("Received query:", query);

    if (!process.env.ASTRA_DB_CLIENT_ID || !process.env.ASTRA_DB_CLIENT_SECRET || !process.env.ASTRA_DB_KEYSPACE || !process.env.ASTRA_DB_SECURE_BUNDLE_PATH) {
      throw new Error('Missing Astra DB environment variables');
    }

    const client = new Client({
      cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
      credentials: { 
        username: process.env.ASTRA_DB_CLIENT_ID,
        password: process.env.ASTRA_DB_CLIENT_SECRET 
      },
      keyspace: process.env.ASTRA_DB_KEYSPACE,
    });

    const prompt = ChatPromptTemplate.fromTemplate(
      `Answer the user's question from the following context: 
      {context}
      Question: {input}`
    );

    await client.connect();
    const embeddings = new OpenAIEmbeddings();
    const retriever = new CustomAstraDBRetriever(client, embeddings);
    const model = new ChatOpenAI({
      openAIApiKey: process.env.OPENAI_API_KEY, 
      modelName: "gpt-3.5-turbo-1106",
      temperature: 0.7,
      maxTokens: 1000,
      maxRetries: 5,
    });

    const combineDocsChain = await createStuffDocumentsChain({ llm: model as any, prompt : prompt});
    const chain = await createRetrievalChain({ retriever, combineDocsChain });
    const response = await chain.invoke({ input: query });

    console.log("Raw response from chain.invoke:", response);

    // Check if response is an object and has a 'text' property
    if (response && typeof response === 'object') {
      const responseText = response.text || response.answer || response.content; // Adjust based on actual structure
      if (responseText) {
        console.log("Generated response:", responseText);
        res.status(200).json({ response: responseText });
      } else {
        console.error("Response is missing expected text property");
        res.status(500).json({ message: 'Failed to generate a response' });
      }
    } else {
      console.error("Response is undefined or not an object");
      res.status(500).json({ message: 'Failed to generate a response' });
    }

    await client.shutdown();
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
  }
} 