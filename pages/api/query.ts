import { NextApiRequest, NextApiResponse } from 'next';
import { OpenAI } from '@langchain/openai';
import { Client } from 'cassandra-driver';
import { OpenAIEmbeddings } from '@langchain/openai';
import { VectorStore } from '@langchain/core/vectorstores';
import { RetrievalQAChain } from 'langchain/chains';

class CustomAstraDBVectorStore extends VectorStore {
  private client: Client;

  constructor(embeddings: OpenAIEmbeddings, client: Client) {
    super(embeddings, {});
    this.client = client;
  }

  async similaritySearch(query: string, k: number) {
    // Implement similarity search using AstraDB
    // This is a placeholder and needs to be implemented based on your AstraDB setup
    const embedding = await this.embeddings.embedQuery(query);
    const queryText = 'SELECT * FROM aviation_documents ORDER BY embedding ANN OF ? LIMIT ?';
    const results = await this.client.execute(queryText, [embedding, k], { prepare: true });
    
    return results.rows.map((row: any) => ({
      pageContent: row.text,
      metadata: { filename: row.filename, chunk_id: row.chunk_id },
    }));
  }
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    const { query } = req.body;

    // Check if environment variables are set
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
    const vectorStore = new CustomAstraDBVectorStore(embeddings, client);

    const model = new OpenAI({ openAIApiKey: process.env.OPENAI_API_KEY });

    const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

    const response = await chain.call({ query });

    await client.shutdown();

    res.status(200).json({ response: response.text });
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ message: 'Internal server error', error: error.message });
  }
}