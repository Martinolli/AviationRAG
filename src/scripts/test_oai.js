import openai from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const configuration = new openai.Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openaiApi = new openai.OpenAIApi(configuration);

console.log('Configuration and OpenAIApi created successfully!');
import openai from 'openai';
import dotenv from 'dotenv';
dotenv.config();

const configuration = new openai.Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});

const openaiApi = new openai.OpenAIApi(configuration);

console.log('Configuration and OpenAIApi created successfully!');
