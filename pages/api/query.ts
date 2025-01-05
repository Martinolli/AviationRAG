import { NextApiRequest, NextApiResponse } from 'next';
import { OpenAI } from '@langchain/openai';
import { Client } from 'cassandra-driver';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { BaseRetriever } from "@langchain/core/retrievers";


class CustomAstraDBRetriever extends BaseRetriever {
  lc_namespace: string[];
  private client: Client;
  private embeddings: OpenAIEmbeddings;

  constructor(client: Client, embeddings: OpenAIEmbeddings) {
    super();
    this.client = client;
    this.embeddings = embeddings;
  }

  async getRelevantDocuments(query: string): Promise<Document[]> {
    const embedding = await this.embeddings.embedQuery(query);
    const queryText = 'SELECT * FROM aviation_documents LIMIT 100'; // Retrieve a batch of documents
    const results = await this.client.execute(queryText, [], { prepare: true });

    // Calculate similarity in application logic
    const documents = results.rows.map((row: any) => new Document({
      pageContent: row.text,
      metadata: { filename: row.filename, chunk_id: row.chunk_id },
    }));

    // Sort documents by similarity
    const sortedDocuments = documents.sort((a, b) => {
      const simA = this.calculateSimilarity(embedding, a);
      const simB = this.calculateSimilarity(embedding, b);
      return simB - simA;
    });

    return sortedDocuments.slice(0, 5); // Return top 5 most similar documents
  }

  private calculateSimilarity(queryEmbedding: number[], document: Document): number {
    // Implement your similarity calculation logic here
    return 0; // Placeholder
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { query } = req.body;

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

    await client.connect();

    const embeddings = new OpenAIEmbeddings();
    const retriever = new CustomAstraDBRetriever(client, embeddings);

    const model = new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });

    const chain = RetrievalQAChain.fromLLM(model, retriever);

    const response = await chain.call({ query });

    await client.shutdown();

    res.status(200).json({ response: response.text });
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
  }
}