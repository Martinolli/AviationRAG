
# Aviation RAG Project

## Aviation RAG Purpose - Introduction

### The aeronautical community and industry are complex. The journey from a simple idea to design, manufacture, assembly, and flight is long. Today, the industry requires connections between different countries and resources and relies on electronic communication to keep this system functioning accurately. The amount of data, information, and decisions involved in the initial stages of a project through to its final release is substantial, significantly influencing safety outcomes. In aviation, safety is not just a set of practices but a reflection of behaviors and culture; it must be integrated into the core of aviation thinking. Given the intricate landscape of components, systems, and regulations, fostering a robust safety culture is crucial. As the extent of data, processes, and information continues to expand, decision-making becomes more complex. However, recent advancements in Large Language Models (LLMs) have enhanced our ability to manage and analyze vast amounts of information. These models find application across various sectors, including healthcare, education and training, cybersecurity, financial services, military operations, and human resources, ultimately leading to improved decision-making and outcomes. Large Language Models (LLMs) are not just tools for marketing and sales; they are capable of understanding language on a massive scale. These models utilize extensive training processes to absorb vast amounts of text, allowing them to predict sentences effectively. LLMs have a wide range of applications and are transforming how we enhance human productivity, particularly in strategic thinking that requires the integration of analyses from various fields. In the aviation industry, for example, LLMs can play a pivotal role during the early stages of system development. They help address critical questions related to materials, systems, suppliers, assembly line methods, operational environments, and other essential variables that must be considered to finalize the design and define the system. Numerous issues may arise throughout this process that require systematic solutions. It's important to emphasize that human performance is a crucial element of this endeavor. We must remember that all processes are interconnected, and human performance plays a crucial role in determining the best ways to achieve goals. In this context, problems should be analyzed with the understanding that every part is linked and that there are no insignificant variables that can be overlooked. Addressing all relevant factors is essential to ensure the safety of system operations in both the civil aviation industry and the military aviation or defense sectors. With this perspective in mind, an important theory that enhances our understanding is the model developed by Mr. James Reason known as the Swiss Cheese Model. This model was later adapted by the Department of Defense (DoD) and led to the creation of the HFACS taxonomy. This framework takes an organizational approach to identify errors and problems, considering the organization as well as the environment in which this complex system operates. It emphasizes the interplay between human decision-making and performance. In this context, the idea of using a large language model (LLM) tool is to support the identification of gaps or deficiencies in this complex system. It acts as an auxiliary to human performance, helping to address issues that ensure outcomes comply with requirements and identify hidden triggers that could potentially lead to accidents or mishaps in the future, whether in the short or long term. This involves analyzing a non-conformance report related to a specific standard to verify if a task is well-defined and to check if the system functions properly during the early stages of design and testing. The use of an LLM model from the top down could be a breakthrough in the aviation industry, considering the actual stage of development necessary for safety operations. The LLM could be the tool used to bring the system safety analysis to the industry level, to treat the main processes as the engineering system treatment. The main objective of this project is to develop a prototype for an LLM tool tailored for the aviation industry. This tool aims to assist managers in evaluating possible alternatives to address initial problems while considering the impact on the final product. It will provide managers with a holistic perspective, as required by the industry's increasing complexity

## Complete AviationRAG Project Description

### Step 1: Define Your Project Goals

- **Goal**: Build a RAG system that allows users to query your aviation corpus in real-time and retrieve relevant information enhanced by an LLM (OpenAI).
- **Main Features**: The ability to search for related documents, generate answers, and augment responses using embedded knowledge from your aviation corpus.
- **Tools**: LangChain.js, Vercel for deployment, Astra DB for vector storage, and OpenAI API for LLM capabilities.

### Step 2: Set Up Your Environment

1. Prerequisites**
   - Install **Node.js** and **npm**: Required for running JavaScript-based LangChain and Vercel.
   - Create an account on **Astra DB** (DataStax) and **Vercel**.
   - Obtain an **OpenAI API key**.

2. Install Necessary Tools**
   - Install LangChain.js: This library will help create embeddings, perform chunking, and connect all components
   - npm install langchain
   - Install Astra DB SDK and Vercel CLI for deploying.
   - npm install @datastax/astra-db-ts vercel

### Step 3: Prepare the Aviation Corpus for Use

1. Process the Corpus into Embeddings**
   - Use LangChain.js to convert each text document into **embeddings**.
   - Preprocess your aviation corpus by breaking documents into chunks suitable for embedding (e.g., 512 or 1024 tokens).
   - Generate embeddings with **OpenAI API** or another compatible embedding provider.
   - javascript
   - const { OpenAIEmbeddings } = require('langchain');
   - const embeddings = await OpenAIEmbeddings.embed(textChunks, { apiKey: 'your_openai_api_key' });

2. Store Embeddings in Astra DB**
   - Connect to Astra DB and create a table for storing documents and their corresponding **vector embeddings**.
   - Store metadata like document titles and keywords to facilitate faster searches.
   - javascript
   - const { Client } = require('@datastax/astra-db-ts');
   - const client = new Client({
       username: 'your_username',
       password: 'your_password',
       secureConnectBundle: 'path/to/secure-connect-database.zip'
   });
   // Store embedding vectors
   - await client.connect();
   - await client.execute('INSERT INTO aviation_data (id, text, embedding) VALUES (?, ?, ?)', [docId, textChunk, embedding]);

### Step 4: Build Retrieval Mechanism

1. Search Embeddings for Relevant Texts**
   - Use **vector similarity search** to locate the most relevant chunks of text when the user asks a question.
   - Astra DB allows you to run similarity queries directly on your embedded vectors.
   - javascript
   - const results = await client.execute('SELECT * FROM aviation_data WHERE similarity(embedding, ?) > threshold', [userQueryEmbedding]);

2. Integrate with OpenAI for RAG**
   - Use **LangChain** to construct a retrieval-augmented generation workflow.
   - The retrieved chunks can be fed to OpenAI’s `gpt-3.5-turbo` or similar to generate detailed responses.
   - javascript
   - const response = await openai.generate({ prompt: `Here is the context: ${retrievedTexts}.
   - Answer the user query: ${userQuery}`, apiKey: 'your_openai_api_key' });

### Step 5: Deploy the Application with Vercel

1. Create a Simple Frontend**
   - Design a minimal **frontend interface** using JavaScript and HTML to allow users to type questions.
   - Use Vercel to host your application for easy scalability.

2. Deploy Your Application**
   - Configure **Vercel** to deploy the application with one command.
   - sh
   - vercel --prod
   - Ensure your app connects to Astra DB and the OpenAI API.

### Step 6: Test and Iterate

1. Testing Workflow**
   - Query the aviation corpus for typical user questions like "What are the common safety measures for maintenance?"
   - Ensure the retrieved context is correct and that the responses from OpenAI are relevant and informative.

2. Refine Embeddings and Pipeline**
   - Review the relevance of the retrieved documents.
   - Fine-tune embedding creation or chunking sizes if necessary.

### Step 7: Monitor and Improve

1. Set Up Monitoring**
   - Monitor the usage of your application (e.g., which questions users ask most often, response accuracy).
   - Use Astra DB’s **built-in monitoring tools** or services like **Datadog** to track performance.

2. Improve Based on Feedback**
   - Continuously improve the relevance of the retrieval and the quality of the LLM’s response.
   - If certain queries aren't returning valuable results, consider enriching the corpus or adjusting the retrieval parameters.

### Summary

- **Environment Setup**: Node.js, Astra DB, Vercel, OpenAI API.
- **Corpus Preparation**: Process the aviation documents into embeddings using LangChain.js and store them in Astra DB.
- **Build Retrieval Mechanism**: Implement vector similarity searches and integrate OpenAI to generate responses.
- **Deploy**: Use Vercel for easy hosting and scalability.
- **Test and Improve**: Iterate on the RAG system based on testing and user feedback.

## Architecture/Folder Structure Overview

    AviationRAG/
    |
    |
    |
    |_____.dvc/
    |      |___cache/
    |      |___tmp/
    |      |___.gitignore
    |      |___config
    |
    |_____.git/
    |      |___hooks/
    |      |___info/
    |      |___lfs/
    |      |___logs/
    |      |___objects/
    |      |___refs/
    |      |___COMMIT_EDITMSG
    |      |___config
    |      |___description
    |      |___FETCH_HEAD
    |      |___HEAD
    |      |___INDEX
    |      |___ORIG_HEAD
    |
    |
    |_____config/
    |          |___secure-connect-aviation_rag-db
    |
    |_____data/
    |      |___documents/documents to be processed "PDF" or "DOCX" files
    |      |___embeddings/aviation_embeddings.json
    |      |___processed/
    |      |           |________chunked_documents/
    |      |           |                        |_______chunked files for each document processed (json)
    |      |           |________aviation_corpus.json
    |      |           |________aviation_corpus.json.dvc
    |      |
    |      |___processed/ProcessedText/
    |      |                          |______processed texts from PDF and DOCX files (TXT files)
    |      |
    |      |___processed/ProcessedTestExapanded/
    |      |                                  |______texts from PDF and DOCX files (TXT files)
    |      |
    |      |___raw/
    |            |___aviation_corpus.pkl
    |
    |_____logs/
    |        |___aviation_rag_manager.log 
    |
    |
    |_____models/
    |
    |
    |
    |_____node_modules/
    |
    |
    |
    |_____public/
    |          |_____index.html
    |
    |
    |
    |_____src/
    |      |
    |      |___components/
    |      |
    |      |
    |      |___scripts/
    |      |          |____ __pycache__/
    |      |          |
    |      |          |____js_files/
    |      |          |           |_____js script files tests
    |      |          |
    |      |          |____py_files
    |      |          |           |_____python files test scripts
    |      |
    |      |__________check_astradb.js - check the Astra database
    |      |
    |      |__________check_astradb_content.js
    |      |
    |      |__________connect_astra.js - connect Astra database
    |      |
    |      |__________create_table_astra.js - create the AstraDB table
    |      |
    |      |__________generate_embeddings.js - generate embeddings from chunk files
    |      |
    |      |__________store_embeddings_astra.js - store the embeddings AstraDB
    |      |
    |      |__________test_openai.js - test openai connection
    |      |
    |      |__________aviation_chunk_saver.py - create the chunks from aviation corpus
    |      |
    |      |__________aviation_rag_manager.py - to manage the pipeline throughout the flow
    |      |
    |      |__________aviationrag_interface.py - generate an interface streamlit to check the embeddings similarity
    |      |
    |      |__________check_pkl_content.py - to check the pkl file content
    |      |
    |      |__________config.py - config the data stored folders
    |      |
    |      |__________embeddings_similarity.py - check the embeddings similarities
    |      |
    |      |__________embeddings_similarity_verification - check the embeddings answers
    |      |
    |      |__________extract_pkl_to_json.py - extract files from corpus to json format
    |      |
    |      |__________read_documents.py - create the corpus from documents processed
    |      |
    |      |__________utils/
    |
    |
    |_____.dvcignore
    |
    |_____.env
    |
    |_____.gitattributes
    |
    |_____.gitignore
    |
    |_____chunking.txt
    |
    |_____diary.txt
    |
    |_____package.json
    |
    |_____processed_files.json
    |
    |_____README.md
    |
    |_____update_data.bat
    |
    |_____vercel.json

# Routine Algorithm

## Remarks

        The original documents are not stored in the same main folder.
        The original documents are preferable in “DOCX” format.
        If the original documents are in “PDF” or another format, the information is assumed to be passed into “DOCX” format to be processed.
        Information with the following subjects is acceptable such as knowledge source:
        -   Aircraft Certification Regulations from different Aviation authorities, such as FAA, EASA, CAA, etc.
        -   Military Standards with information related to System Safety, System Design, Requirements Criteria, Handbooks, etc.
        Technical books addressing knowledge from the following areas:
        -   Aircraft Design
        -   Aircraft System Safety Analysis
        -   Aircraft Operation
        -   Aircraft Maintenance
        -   Aircraft Certification
        ISO/SAE Standards
        -   Quality
        -   ARP documents
        -   Etc.

## Routine Steps

### Prepare the document to be processed

1. Choose the document from the library and check if it is in a convenient format, preferably "PDF” or “DOCX” format; check the document content, whether it has pictures, tables, and formulas, and whether the text is in column format or not. For the first documents, it is defined that it will accept only documents in “PDF” format and with one-column text
2. If the document is in “PDF” format, change it to “DOCX” format to be processed. This step could be changed in the future, but it was defined for this first prototype to follow this line.
3. Change the document in “DOCX” format or if not necessary, check the document content, number of pages, if it has tables, headers, footers, page numbers, etc. Check the size of documents If they are too extensive, split the document in a better way to process, as recommended by the chapters.

4. After the document is prepared store it in the corresponding folder in the AviationRAG folder

- AviationRAG/data/documents

### Process the document

1. This part of the routine is responsible for preparing the core of the complete process flow, generating the document tokens, and saving the “aviation_corpus.pkl” file, where the data are stored for future manipulation and creation of the embeddings. The routine that implements this task is the “read_documents.py.” Following is the summary of the routine steps
2. Check if there is a new document in the “documents” folder
3. If there is a new document, the process starts.
4. Process the document

    - read the "PDF" or "DOCX" file
    - change it to "TXT" file;
    - save it to ProcessedText folder, data/ProcessedText Folder storage
    - start process the "TXT" file:
        - remove the stop words
        - clean the file
        - extract the text
        - extract the tokens
        - extract names
        - check the abbreviations
        - check the pos-tags
        - check for metadata
        - generate the dictionary with all information about the new corpus
        - generate the aviation_corpus.pkl file
        - store the aviation_corpus.pkl file in the correct folder, aviationrag/data/raw
    - store the information in the dictionary format
        - filename
        - text
        - tokens
        - names
        - entities
        - pos-tags
        - matadata
        - final file = aviation_corpus.pkl
        - stored the file in the data/documents/raw

### Generate the Chunks

1. First Step

2. Second Step

    - alfa
    - beta
    - charlie

3. Thirs Step

### Saved chunks as a json file

### Generate the embeddings for chunks

### Generate the similarity cosines to check the results adherence to the queries

### Store the embeddings into AstraDB

### Generate embeddings questions for test

# Scripts Description - Below the scripts in the flow sequence

## 1 - read_documents.py - First script

This script read documents from data/documents The documents shall be in the following formats: "PDF" or "DOCX" the preferable format is "DOCX"
Output = aviation_corpus.pkl file

    ```python

    import os
    import pickle
    import pdfplumber
    import spacy
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    import csv
    import re
    from docx import Document
    from spellchecker import SpellChecker
    import wordninja
    from sklearn.feature_extraction.text import TfidfVectorizer
    import PyPDF2
    import logging

    # Load spaCy's English model
    nlp = spacy.load('en_core_web_sm')
    nlp.max_length = 2000000  # or any other suitable value
    # Download required NLTK data
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    # Initialize spellchecker
    spell = SpellChecker()
    # Suppress specific warnings
    import warnings
    warnings.filterwarnings("ignore", message="usetex mode requires TeX.")
    # Global stopwords
    STOP_WORDS = set(stopwords.words('english'))

    # Configure logging
    logging.basicConfig(level=logging.INFO, filename='read_documents.log', format='%(asctime)s - %(levelname)s - %(message)s')

    # Define base directory
    BASE_DIR = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'

    # Define paths
    TEXT_OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'ProcessedText')
    TEXT_EXPANDED_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'ProcessedTextExpanded')
    PKL_FILENAME = os.path.join(BASE_DIR, 'data', 'raw', 'aviation_corpus.pkl')

    # Ensure directories exist
    for directory in [TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR, os.path.dirname(PKL_FILENAME)]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Created directory: {directory}")
        else:
            logging.info(f"Directory already exists: {directory}")

    # Create custom pipeline component for aviation NER
    @spacy.Language.component("aviation_ner")
    def aviation_ner(doc):
        logging.info(f"Starting aviation_ner for document: {doc[:50]}...")
        patterns = [
            ("AIRCRAFT_MODEL", r"\b[A-Z]-?[0-9]{1,4}\b"),
            ("AIRPORT_CODE", r"\b[A-Z]{3}\b"),
            ("FLIGHT_NUMBER", r"\b[A-Z]{2,3}\s?[0-9]{1,4}\b"),
            ("AIRLINE", r"\b(American Airlines|Delta Air Lines|United Airlines|Southwest Airlines|Air France|Lufthansa|British Airways)\b"),
            ("AVIATION_ORG", r"\b(FAA|EASA|ICAO|IATA)\b"),
        ]

        new_ents = []
        for ent_type, pattern in patterns:
            for match in re.finditer(pattern, doc.text):
                start, end = match.span()
                    span = doc.char_span(start, end, label=ent_type)
                    if span is not None:
                        # Check for overlap with existing entities
                        if not any(span.start < ent.end and span.end > ent.start for ent in list(doc.ents) + new_ents):
                            new_ents.append(span)
                            logging.debug(f"Added new entity: {span.text} ({ent_type})")
                            
            doc.ents = list(doc.ents) + new_ents
            logging.info(f"Finished aviation_ner. Added {len(new_ents)} new entities.")
            return doc

        # Add the custom component to the pipeline
        nlp.add_pipe("aviation_ner", after="ner")

        def load_abbreviation_dict():
            abbreviation_dict = {}
            try:
                with open('abbreviations.csv', mode='r') as infile:
                    reader = csv.reader(infile)
                    for rows in reader:
                        if len(rows) < 2:
                            continue
                        abbreviation_dict[rows[0].strip()] = rows[1].strip()
            except FileNotFoundError:
                print("Error: The file 'abbreviations.csv' was not found.")
            except Exception as e:
                print(f"An error occurred while loading the abbreviation dictionary: {e}")
            return abbreviation_dict

        def split_connected_words_improved(text):
            words = re.findall(r'\w+|\W+', text)
            split_words = []
            for word in words:
                if len(word) > 15 and word.isalnum():
                    split_parts = re.findall('[A-Z][a-z]*|[a-z]+|[0-9]+', word)
                    split_words.extend(split_parts)
                else:
                    split_words.append(word)
            split_words = ' '.join(split_words)
            split_words = ' '.join(wordninja.split(split_words))
            return split_words

        def filter_non_sense_strings(text):
            words = text.split()
            cleaned_words = []
            for word in words:
                if re.match(r'^[a-zA-Z]+$', word) and len(set(word.lower())) > 3:
                    cleaned_words.append(word)
            return ' '.join(cleaned_words)

        def preprocess_text_with_sentences(text):
            doc = nlp(text)
            sentences = []
            for sent in doc.sents:
                cleaned_sentence = ' '.join(
                    token.lemma_.lower() for token in sent
                    if token.is_alpha and token.text.lower() not in STOP_WORDS
                )
                if cleaned_sentence:
                    sentences.append(cleaned_sentence)
            return ' '.join(sentences)

        def extract_personal_names(text):
            doc = nlp(text)
            return [ent.text for ent in doc.ents if ent.label_ == 'PERSON']

        def extract_entities_and_pos_tags(text):
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [(token.text, token.pos_) for token in doc]
            return entities, pos_tags

        def expand_abbreviations_in_text(text, abbreviation_dict):
            words = text.split()
            expanded_words = []
            for word in words:
                if word.lower() in abbreviation_dict:
                    expanded_words.append(abbreviation_dict[word.lower()])
                else:
                    expanded_words.append(word)
            return ' '.join(expanded_words)

        def extract_text_from_pdf_with_pdfplumber(pdf_path):
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = ''.join([page.extract_text() + '\n' for page in pdf.pages])
                    return text
            except Exception as e:
                print(f"Failed to process PDF {pdf_path}: {e}")
                return ""

        def extract_keywords(documents, top_n=10):
            texts = [doc['text'] for doc in documents]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            feature_names = vectorizer.get_feature_names_out()
            for idx, doc in enumerate(documents):
                tfidf_scores = tfidf_matrix[idx].toarray()[0]
                sorted_indices = tfidf_scores.argsort()[::-1]
                doc['keywords'] = [feature_names[i] for i in sorted_indices[:top_n]]

        def extract_metadata(file_path):
            metadata = {}
            if file_path.endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata = reader.metadata
            # Add more file types as needed
            return metadata

        def classify_document(text):
            keywords = {
                'safety': ['safety', 'hazard', 'risk', 'incident', 'accident','system','hazard','emergency'],
                'maintenance': ['maintenance', 'repair', 'overhaul', 'inspection'],
                'operations': ['flight', 'takeoff', 'landing', 'crew', 'pilot','aircraft', 'airplane'],
                'regulations': ['regulation', 'compliance', 'standard', 'rule', 'law'],
                'quality': ['quality', 'performance', 'service', 'customer', 'satisfaction','design'],
            }
            
            text_lower = text.lower()
            scores = {category: sum(1 for word in words if word in text_lower) for category, words in keywords.items()}
            return max(scores, key=scores.get)

        def read_documents_from_directory(directory_path, text_output_dir=None, text_expanded_dir=None, existing_documents=None):
            logging.info(f"Starting to read documents from {directory_path}")
            if existing_documents is None:
                existing_documents = []
            
            existing_files = {doc['filename'] for doc in existing_documents}
            new_documents = []
            abbreviation_dict = load_abbreviation_dict()
            lemmatizer = WordNetLemmatizer()

            for filename in os.listdir(directory_path):
                logging.info(f"Processing file: {filename}")
                if filename in existing_files:
                    continue

                file_path = os.path.join(directory_path, filename)
                text = ''
                if filename.endswith(".pdf"):
                    logging.info(f"Extracting text from PDF: {filename}")
                    text = extract_text_from_pdf_with_pdfplumber(file_path)
                elif filename.endswith(".docx"):
                    logging.info(f"Extracting text from DOCX: {filename}")
                    try:
                        doc = Document(file_path)
                        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                    except Exception as e:
                        logging.error(f"Failed to process DOCX {filename}: {e}")
                        print(f"Failed to process DOCX {filename}: {e}")
                        continue
                else:
                    logging.warning(f"Skipping unsupported file type: {filename}")
                    continue

                logging.info(f"Preprocessing text from {filename}")
                if not text:
                    logging.warning(f"No text extracted from {filename}")
                    continue

                expanded_text = expand_abbreviations_in_text(text, abbreviation_dict)
                raw_text = expanded_text
                expanded_text = split_connected_words_improved(expanded_text)
                expanded_text = filter_non_sense_strings(expanded_text)
                preprocessed_text = preprocess_text_with_sentences(expanded_text)
                personal_names = extract_personal_names(preprocessed_text)
                entities, pos_tags = extract_entities_and_pos_tags(preprocessed_text)
                tokens = word_tokenize(preprocessed_text)
                cleaned_tokens = [token.lower() for token in tokens if token.isalpha() and len(token) > 2]
                tokens_without_stopwords = [token for token in cleaned_tokens if token not in STOP_WORDS]
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_stopwords]

                if text_expanded_dir:
                    output_file_path = os.path.join(text_expanded_dir, f'{filename}.txt')
                    with open(output_file_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(raw_text)
                    logging.info(f"Expanded text saved to: {output_file_path}")
                    print(f"Expanded text saved to: {output_file_path}")
                    
                if text_output_dir:
                    output_file_path = os.path.join(text_output_dir, f'{filename}.txt')
                    logging.info(f"Processed text saved to: {output_file_path}")
                    with open(output_file_path, 'w', encoding='utf-8') as out_file:
                        out_file.write(preprocessed_text)
                    print(f"Text saved to: {output_file_path}")

                logging.info(f"Finished processing all documents in {directory_path}")
                metadata = extract_metadata(file_path)
                document_category = classify_document(preprocessed_text)

                new_documents.append({
                    'filename': filename,
                    'text': preprocessed_text,
                    'tokens': lemmatized_tokens,
                    'personal_names': personal_names,
                    'entities': entities,
                    'pos_tags': pos_tags,
                    'metadata': metadata,
                    'category': document_category
                })

            return existing_documents + new_documents

        def update_existing_documents(documents):
            for doc in documents:
                if 'metadata' not in doc:
                    doc['metadata'] = extract_metadata(os.path.join(BASE_DIR, doc['filename']))
                if 'category' not in doc:
                    doc['category'] = classify_document(doc['text'])
            return documents

        def main():
            documents = None
            if os.path.exists(PKL_FILENAME):
                with open(PKL_FILENAME, 'rb') as file:
                    documents = pickle.load(file)
                documents = update_existing_documents(documents)

            if documents is None:
                print("Reading documents from directory...")
                documents = read_documents_from_directory(BASE_DIR, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR)
            else:
                print("Appending new documents to the existing list...")
                documents = read_documents_from_directory(BASE_DIR, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR, documents)

            # Apply keyword extraction
            extract_keywords(documents)

            # Save the updated list
            with open(PKL_FILENAME, 'wb') as file:
                pickle.dump(documents, file)

            print(f"Total documents: {len(documents)}")

        if __name__ == '__main__':
            logging.info("Starting document processing script")
            main()
            logging.info("Document processing script completed")
    ```

## aviation_chunk_saver.py

    - This script creates the chunks from data/raw/aviation_corpus.pkl file
    - Output = chunks for each document from aviation_corpus.pkl stored in the data/processed/chunked_documents

    ```python

    import os
    import json
    import logging
    import nltk
    from nltk.tokenize import sent_tokenize
    import tiktoken
    import pickle

    # Ensure necessary NLTK data is downloaded
    nltk.download('punkt')

    # Define absolute paths
    base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
    pkl_file = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
        chunk_output_dir = os.path.join(base_dir, 'data', 'processed', 'chunked_documents')

        # Set up logging
        logging.basicConfig(level=logging.INFO, filename='chunking.log',
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Directory to save chunked JSON files
        if not os.path.exists(chunk_output_dir):
            os.makedirs(chunk_output_dir)

        # Initialize OpenAI tokenizer for accurate token counting
        tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

        # Function to count tokens using OpenAI's tokenizer
        def count_tokens(text):
            return len(tokenizer.encode(text))

        # Function to chunk text by sentences and enforce token limits
        def chunk_text_by_sentences(text, max_tokens=500, overlap=50):
            sentences = sent_tokenize(text)  # Tokenize into sentences
            chunks = []
            current_chunk = []
            current_tokens = 0

            for sentence in sentences:
                sentence_token_count = count_tokens(sentence)

                # Check if adding this sentence exceeds the max token limit
                if current_tokens + sentence_token_count > max_tokens:
                    # Save the current chunk only if it's not empty
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    # Start a new chunk with overlap (only if not the first chunk)
                    current_chunk = current_chunk[-overlap:] if overlap and len(chunks) > 0 else []
                    current_tokens = count_tokens(" ".join(current_chunk))

                current_chunk.append(sentence)
                current_tokens += sentence_token_count

            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Validate and split oversized chunks
            return validate_and_split_chunks(chunks, max_tokens)

        # Function to validate and split oversized chunks
        def validate_and_split_chunks(chunks, max_tokens):
            """Ensure all chunks are within the token limit."""
            validated_chunks = []
            for chunk in chunks:
                token_count = count_tokens(chunk)
                if token_count > max_tokens:
                    logging.warning(f"Chunk exceeds token limit: {token_count} tokens. Splitting further.")
                    # Split the chunk into smaller parts
                    words = chunk.split()
                    temp_chunk = []
                    temp_tokens = 0
                    for word in words:
                        word_token_count = count_tokens(word)
                        if temp_tokens + word_token_count > max_tokens:
                            validated_chunks.append(" ".join(temp_chunk))
                            temp_chunk = []
                            temp_tokens = 0
                        temp_chunk.append(word)
                        temp_tokens += word_token_count
                    if temp_chunk:
                        validated_chunks.append(" ".join(temp_chunk))
                else:
                    validated_chunks.append(chunk)
            return validated_chunks

        # Function to process documents and save chunks as JSON
        def save_documents_as_chunks(documents, output_dir, max_tokens=500, overlap=50):
            for doc in documents:
                filename = doc['filename']
                text = doc['text']
                metadata = doc.get('metadata', {})  # Get metadata if it exists, otherwise empty dict
                category = doc.get('category', '')  # Get category if it exists, otherwise empty string

                chunks = chunk_text_by_sentences(text, max_tokens, overlap)
                validated_chunks = validate_and_split_chunks(chunks, max_tokens)

                output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_chunks.json")
                
                chunk_data = {
                    "filename": filename,
                    "metadata": metadata,
                    "category": category,
                    "chunks": [
                        {
                            "text": chunk,
                            "tokens": count_tokens(chunk)
                        } for chunk in validated_chunks
                    ]
                }

                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)

                logging.info(f"Processed and saved chunks for {filename}")

        # Main routine
        def main():
            # Load your PKL file containing documents
            if not os.path.exists(pkl_file):
                logging.error(f"Error: PKL file '{pkl_file}' not found!")
                return

            try:
                with open(pkl_file, 'rb') as file:
                    documents = pickle.load(file)
                logging.info(f"Loaded {len(documents)} documents.")
            except Exception as e:
                logging.error(f"Failed to load PKL file: {e}")
                return

            # Process and save chunks for all documents
            save_documents_as_chunks(documents, chunk_output_dir)

            logging.info(f"All documents processed. Chunks saved in '{chunk_output_dir}'.")

        if __name__ == '__main__':
            main()
    ```

## extract_pkl_to_json.py

This script extract the original aviation_corpus.pkl to json format and store it on the aviation_corpus.json file

    ```python

        import pickle
        import json
        import os

        # Define absolute paths
        base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
        pkl_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
        json_path = os.path.join(base_dir, 'data', 'processed', 'aviation_corpus.json')

        # Load the pickle file
        def extract_pkl_to_json(pkl_path, json_path):
            with open(pkl_path, 'rb') as file:
                corpus = pickle.load(file)
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(corpus, json_file, ensure_ascii=False, indent=4)

        print(f"Data successfully extracted and saved to {json_path}")
    ```

## generate_embeddings.js

This script generates the embeddings from chunked_documents
output - embeddings from chunks saved in the data/embeddings

    ```java
        const fs = require('fs');
        const path = require('path');
        const dotenv = require('dotenv');
        const { Configuration, OpenAIApi } = require('openai');

        // Load environment variables
        dotenv.config();

        const configuration = new Configuration({
        apiKey: process.env.OPENAI_API_KEY,
        });
        const openai = new OpenAIApi(configuration);

        // Utility function to add delay
        function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
        }

        // Function to load existing embeddings
        function loadExistingEmbeddings(outputPath) {
        if (fs.existsSync(outputPath)) {
            const rawData = fs.readFileSync(outputPath, 'utf-8');
            return JSON.parse(rawData);
        }
        return [];
        }

        // Function to check if chunk ID exists
        function isChunkIdExists(existingEmbeddings, chunk_id) {
        return existingEmbeddings.some(embedding => embedding.chunk_id === chunk_id);
        }

        // Function to process a single chunk
        async function processChunk(chunk, filename, index, existingEmbeddings) {
        const chunk_id = `${filename}-${index}`; // Generate chunk ID dynamically

        // Skip if the chunk ID already exists
        if (isChunkIdExists(existingEmbeddings, chunk_id)) {
            console.log(`Skipping duplicate chunk ID: ${chunk_id}`);
            return null;
        }

        let attempts = 0;
        const maxAttempts = 3;

        while (attempts < maxAttempts) {
            try {
            const response = await openai.createEmbedding({
                model: 'text-embedding-ada-002',
                input: chunk.text,
            });

            const embeddingVector = response.data.data[0].embedding;
            console.log(`Generated embedding for chunk ID: ${chunk_id}`);
            return {
                chunk_id: chunk_id,
                filename: filename,
                text: chunk.text,
                tokens: chunk.tokens, // Add tokens count for reference
                embedding: embeddingVector,
            };
            } catch (err) {
            attempts++;
            console.error(`Error generating embedding for chunk ID: ${chunk_id} (Attempt ${attempts})`, err);
            if (attempts >= maxAttempts) {
                console.error(`Failed to generate embedding for chunk ID: ${chunk_id} after ${maxAttempts} attempts`);
                return null;
            }
            await delay(2000); // Wait before retrying
            }
        }
        }

        // Function to process files from the chunked documents directory
        async function processFile(filePath, existingEmbeddings) {
        const rawData = fs.readFileSync(filePath, 'utf-8');
        const chunkedDoc = JSON.parse(rawData);
        const filename = chunkedDoc.filename;
        const category = chunkedDoc.category; // Metadata
        const embeddings = [];

        console.log(`Processing file: ${filename}, Category: ${category}`);
        for (let i = 0; i < chunkedDoc.chunks.length; i++) {
            const chunk = chunkedDoc.chunks[i];
            const result = await processChunk(chunk, filename, i, existingEmbeddings);
            if (result) embeddings.push(result);
            await delay(500); // Delay between processing chunks
        }

        return embeddings;
        }

        // Main function to generate embeddings
        async function generateEmbeddings() {
        try {
            const chunkedDocsPath = path.join(__dirname, '../../data/processed/chunked_documents');
            const outputPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
            const files = fs.readdirSync(chunkedDocsPath).filter(file => file.endsWith('.json'));

            // Load existing embeddings
            let allEmbeddings = loadExistingEmbeddings(outputPath);

            console.log(`Found ${files.length} files to process.`);
            for (const file of files) {
            const filePath = path.join(chunkedDocsPath, file);
            const embeddings = await processFile(filePath, allEmbeddings);
            allEmbeddings = allEmbeddings.concat(embeddings);
            }

            // Save all embeddings to a JSON file
            await fs.promises.writeFile(outputPath, JSON.stringify(allEmbeddings, null, 2));
            console.log(`Embeddings saved to ${outputPath}`);
        } catch (err) {
            console.error('Error while generating embeddings:', err);
        }
        }

        // Run the function
        generateEmbeddings();
    ```

## store_embeddings_astra.js

store the embeddings in the AstraDB: aviation_rag_db/aviation_data/aviation_documents

    ```java
        const cassandra = require('cassandra-driver');
        const fs = require('fs').promises;
        const path = require('path');
        const dotenv = require('dotenv');

        dotenv.config();

        async function insertEmbeddings() {
            const client = new cassandra.Client({
                cloud: { secureConnectBundle: process.env.ASTRA_DB_SECURE_BUNDLE_PATH },
                credentials: {
                    username: process.env.ASTRA_DB_CLIENT_ID,
                    password: process.env.ASTRA_DB_CLIENT_SECRET,
                },
                keyspace: process.env.ASTRA_DB_KEYSPACE,
            });

            try {
                await client.connect();
                console.log('Connected to Astra DB');

                const embeddingsPath = path.join(__dirname, '../../data/embeddings/aviation_embeddings.json');
                const embeddingsData = JSON.parse(await fs.readFile(embeddingsPath, 'utf8'));

                const selectQuery = 'SELECT chunk_id FROM aviation_documents WHERE chunk_id = ?';
                const insertQuery = 'INSERT INTO aviation_documents (chunk_id, filename, text, tokens, embedding) VALUES (?, ?, ?, ?, ?)';

                for (const item of embeddingsData) {
                    // Check if the chunk_id already exists
                    const result = await client.execute(selectQuery, [item.chunk_id], { prepare: true });

                    if (result.rows.length > 0) {
                        console.log(`Skipping chunk_id: ${item.chunk_id} (already exists)`);
                    } else {
                        // Convert the embedding array to a Buffer
                        const embeddingBuffer = Buffer.from(new Float32Array(item.embedding).buffer);

                        // Insert new embedding
                        await client.execute(insertQuery, [
                            item.chunk_id,
                            item.filename,
                            item.text,
                            item.tokens,
                            embeddingBuffer
                        ], { prepare: true });
                        console.log(`Inserted embedding for chunk_id: ${item.chunk_id}`);
                    }
                }

                console.log('All embeddings processed successfully');
            } catch (err) {
                console.error('Error:', err);
            } finally {
                await client.shutdown();
            }
        }

        insertEmbeddings();
    ```

## Supportive Routines

The following routines were created to:

Manage the flow since from "read" a document and create the chunks to verify the similarities and check the files contents
Check the AstraDB content
Create the AstraDB table
Connect AstraDB database
