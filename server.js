const express = require('express');
const dotenv = require('dotenv');
const { AstraClient } = require('@astrajs/collections');
const { Configuration, OpenAIApi } = require('openai');

dotenv.config();

const app = express();
app.use(express.json());

const astraClient = new AstraClient({
  astraDatabaseId: process.env.ASTRA_DB_ID,
  astraDatabaseRegion: process.env.ASTRA_DB_REGION,
  applicationToken: process.env.ASTRA_DB_APPLICATION_TOKEN,
});

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

app.post('/query', async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'Query not provided' });
  }

  try {
    const collection = astraClient.namespace('aviation_data').collection('aviation_documents');
    const documents = await collection.find({});

    const response = await openai.createCompletion({
      model: 'text-davinci-003',
      prompt: query,
      max_tokens: 150,
      temperature: 0.7,
    });

    res.status(200).json({ response: response.data.choices[0].text });
  } catch (error) {
    console.error('Error processing query:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});