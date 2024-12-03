# Aviation RAG Projec

## Aviation RAG Purpose - Introduction

### The aeronautical community and industry are complex. The journey from a simple idea to design, manufacture, assembly, and flight is long. Today, the industry requires connections between different countries and resources and relies on electronic communication to keep this system functioning accurately. The amount of data, information, and decisions involved in the initial stages of a project through to its final release is substantial, significantly influencing safety outcomes. In aviation, safety is not just a set of practices but a reflection of behaviors and culture; it must be integrated into the core of aviation thinking. Given the intricate landscape of components, systems, and regulations, fostering a robust safety culture is crucial. As the extent of data, processes, and information continues to expand, decision-making becomes more complex. However, recent advancements in Large Language Models (LLMs) have enhanced our ability to manage and analyze vast amounts of information. These models find application across various sectors, including healthcare, education and training, cybersecurity, financial services, military operations, and human resources, ultimately leading to improved decision-making and outcomes. Large Language Models (LLMs) are not just tools for marketing and sales; they are capable of understanding language on a massive scale. These models utilize extensive training processes to absorb vast amounts of text, allowing them to predict sentences effectively. LLMs have a wide range of applications and are transforming how we enhance human productivity, particularly in strategic thinking that requires the integration of analyses from various fields. In the aviation industry, for example, LLMs can play a pivotal role during the early stages of system development. They help address critical questions related to materials, systems, suppliers, assembly line methods, operational environments, and other essential variables that must be considered to finalize the design and define the system. Numerous issues may arise throughout this process that require systematic solutions. It's important to emphasize that human performance is a crucial element of this endeavor. We must remember that all processes are interconnected, and human performance plays a crucial role in determining the best ways to achieve goals. In this context, problems should be analyzed with the understanding that every part is linked and that there are no insignificant variables that can be overlooked. Addressing all relevant factors is essential to ensure the safety of system operations in both the civil aviation industry and the military aviation or defense sectors. With this perspective in mind, an important theory that enhances our understanding is the model developed by Mr. James Reason known as the Swiss Cheese Model. This model was later adapted by the Department of Defense (DoD) and led to the creation of the HFACS taxonomy. This framework takes an organizational approach to identify errors and problems, considering the organization as well as the environment in which this complex system operates. It emphasizes the interplay between human decision-making and performance. In this context, the idea of using a large language model (LLM) tool is to support the identification of gaps or deficiencies in this complex system. It acts as an auxiliary to human performance, helping to address issues that ensure outcomes comply with requirements and identify hidden triggers that could potentially lead to accidents or mishaps in the future, whether in the short or long term. This involves analyzing a non-conformance report related to a specific standard to verify if a task is well-defined and to check if the system functions properly during the early stages of design and testing. The use of an LLM model from the top down could be a breakthrough in the aviation industry, considering the actual stage of development necessary for safety operations. The LLM could be the tool used to bring the system safety analysis to the industry level, to treat the main processes as the engineering system treatment. The main objective of this project is to develop a prototype for an LLM tool tailored for the aviation industry. This tool aims to assist managers in evaluating possible alternatives to address initial problems while considering the impact on the final product. It will provide managers with a holistic perspective, as required by the industry's increasing complexity

## Overview

### Step 1: Define Your Project Goals

- Goal: Build a RAG system that allows users to query your aviation corpus in real-time and retrieve relevant information enhanced by an LLM (OpenAI).

- Main Features: The ability to search for related documents, generate answers, and augment responses using embedded knowledge from your aviation corpus.

- Tools: LangChain.js, Vercel for deployment, Astra DB for vector storage, and OpenAI API for LLM capabilities.

### Step 2: Set Up Your Environment

#### Prerequisites

- Install Node.js and npm: Required for running JavaScript-based LangChain and Vercel.

- Create an account on Astra DB (DataStax) and Vercel.

- Obtain an OpenAI API key.

#### Install Necessary Tools

- Install LangChain.js: This library will help create embeddings, perform chunking, and connect all components.
- npm install langchain
- Install Astra DB SDK and Vercel CLI for deploying.
- npm install @datastax/astra-db-ts vercel

### Step 3: Prepare the Aviation Corpus for Use

#### Process the Corpus into Embeddings

- Use LangChain.js to convert each text document into embeddings.

- Preprocess your aviation corpus by breaking documents into chunks suitable for embedding (e.g., 512 or 1024 tokens).

- Generate embeddings with OpenAI API or another compatible embedding provider.

- const { OpenAIEmbeddings } = require('langchain');
- const embeddings = await OpenAIEmbeddings.embed(textChunks, { apiKey: 'your_openai_api_key' });

#### Store Embeddings in Astra DB

- Connect to Astra DB and create a table for storing documents and their corresponding vector embeddings.

- Store metadata like document titles and keywords to facilitate faster searches.

- const { Client } = require('@datastax/astra-db-ts');
- const client = new Client({
    username: 'your_username',
    password: 'your_password',
    secureConnectBundle: 'path/to/secure-connect-database.zip'
});
- // Store embedding vectors
- await client.connect();
 -await client.execute('INSERT INTO aviation_data (id, text, embedding) VALUES (?, ?, ?)', [docId, textChunk, embedding]);

### Step 4: Build Retrieval Mechanism

#### Search Embeddings for Relevant Texts

- Use vector similarity search to locate the most relevant chunks of text when the user asks a question.

- Astra DB allows you to run similarity queries directly on your embedded vectors.

- const results = await client.execute('SELECT * FROM aviation_data WHERE similarity(embedding, ?) > threshold', [userQueryEmbedding]);

#### Integrate with OpenAI for RAG

- Use LangChain to construct a retrieval-augmented generation workflow.

- The retrieved chunks can be fed to OpenAI’s gpt-3.5-turbo or similar to generate detailed responses.

- const response = await openai.generate({ prompt: `Here is the context: ${retrievedTexts}.
- Answer the user query: ${userQuery}`, apiKey: 'your_openai_api_key' });

### Step 5: Deploy the Application with Vercel

#### Create a Simple Frontend

- Design a minimal frontend interface using JavaScript and HTML to allow users to type questions.

- Use Vercel to host your application for easy scalability.

#### Deploy Your Application

- Configure Vercel to deploy the application with one command.

- vercel --prod

- Ensure your app connects to Astra DB and the OpenAI API.

### Step 6: Test and Iterate

#### Testing Workflow

- Query the aviation corpus for typical user questions like "What are the common safety measures for maintenance?"

- Ensure the retrieved context is correct and that the responses from OpenAI are relevant and informative.

#### Refine Embeddings and Pipeline

- Review the relevance of the retrieved documents.

- Fine-tune embedding creation or chunking sizes if necessary.

### Step 7: Monitor and Improve

#### Set Up Monitoring

- Monitor the usage of your application (e.g., which questions users ask most often, response accuracy).

- Use Astra DB’s built-in monitoring tools or services like Datadog to track performance.

#### Improve Based on Feedback

- Continuously improve the relevance of the retrieval and the quality of the LLM’s response.

- If certain queries aren't returning valuable results, consider enriching the corpus or adjusting the retrieval parameters.

### Summary

1. Environment Setup: Node.js, Astra DB, Vercel, OpenAI API.

2. Corpus Preparation: Process the aviation documents into embeddings using LangChain.js and store them in Astra DB.

3. Build Retrieval Mechanism: Implement vector similarity searches and integrate OpenAI to generate responses.

4. Deploy: Use Vercel for easy hosting and scalability.

5. Test and Improve: Iterate on the RAG system based on testing and user feedback.

## Aviation RAG Project Composition

    The functionalities are distributed under two main blocks; AviationDOC scripts, and AviationRAG scripts and folders. These folders are described below.

### The ARAG is composed of several different elements

1. AviationDOC folder
2. AviationRAG folder

#### AviationDOC Folder

    The ADOC Folder contains the following scripts and functions:
    - Chunked_documents Folder - where the chunked documents are stored to be used by the ARAG.
    - Data Folder - where the documents are stored, the documents are the raw data in "PDF" or "DOC" format to be treated and become part of "documents.pkl file" as a corpus document.
    - env Folder.
    - ProcessedText Folder - where the processed texts are stored.
    - Abbreviations.csv file - contains the abbrevations processed during the documents processing.
    - aviation_chunk_saver.py script - this script creates chunks from the processed documents by read_documents_to_txt.py script.
    - chunking.log File.
    - documents.pkl File - contains all documents processed by read_documents_to_txt.py script.
    - read_documents_to_txt script - this script reads the documents stored in the Data Folder, transform them into "TXT" file, clean the documents, create tokens, and prepare them for the chunk processing.
    - requirements.txt file - sotore the python packages required for the processing routines.
    - Data_1 Folder - contains documents to be processed before they will be stored in the Data Folder, for a previously analysis of size and characteristics.

#### AviationRAG Folder

    The ARAG folder contains the following scripts and functions:
    - config folder - contains keys and values necessary to run the condes.
    - data folder - contains the following sub folders:
    1. embeddings folder - contains the embeddings for each document.
    2. processed folder - contains the sub folder with chunked documents, from ADOC subroutines, and aviation corpus.json file from documenst.pkl file in the ADOC folder.
    3. raw folder - contains aviation_corpus.pkl = documents.pkl file, from the original aviation documents.
    - models folder - trained models or LLM configuration files specific for the application. 
    - node_modules folder - where the modules necessary to run the application are located and stored.
    - public folder - where static assets are stored, that are accessible to users when the app is running, for instance; index.html
    - src folder - where all JavaScript files are stored, and components.
    1. components folder - reusable UI components or specific scripts for the front end.
    2. scripts folder - general scripts that handle backend logic such as retrievel, embedding, or database queries.
    3. utils folder - store the functions that might be used across different parts of the project.
    - .env folder - stores API keys and sensitive information, such as OpenAI keys, AstraDB credentials.
    - .gitignore folder - used to list files and directoris that shouldn't be added to the version control, for instance; node_modules, .env, data/raw if the documents are sensitive.
    - package.json and package-lock.json - these files manage the dependencies for the node.js environment and ensure that consitency of versions across the development environment is maintained.
    - README.md - Help documentation for the project.
    - vercel.json - deployment configuration, for deployment of specific configurations, setting up API routing or environment settings.

## Pseudo Code

## Issues Description

### AviationDOC issues

1. Document reading and preprocessing.

- Reading the documentation seems to be the most important part of the project.

### AviationRAG issues
