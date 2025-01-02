const { Client } = require('@datastax/astra');
const { OpenAI } = require('langchain');

const client = new Client({
  secureConnectBundle: './config/secure-connect-database.zip',
});

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'Query not provided' });
  }

  try {
    // Retrieve relevant chunks from AstraDB
    const rows = await client.execute(
      'SELECT text FROM aviation_data.aviation_documents WHERE ...',
      // Add your query conditions to retrieve embeddings or chunks
    );

    const documents = rows.map(row => row.text);

    // Generate a response with LangChain
    const response = await openai.call({
      input: query,
      documents,
    });

    res.status(200).json({ response });
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
