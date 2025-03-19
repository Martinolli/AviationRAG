# Aviation Scraping Plan

AviationRAG Web Scraping Development Plan

## **1. Current Status of the AviationRAG Project**

- **Pipeline Implementation**: `aviationrag_manager.py` is effectively managing the document processing pipeline.
- **Chat System Stability**: `aviationai.py` is functional, and while future adaptations may be needed, it is currently working well.
- **Search Optimization**: FAISS has been implemented for efficient and comprehensive similarity search, significantly improving retrieval speed and accuracy.
- **Testing Phase**: FAISS is undergoing tests to validate its performance with the stored embeddings.

## **Actual Pipeline

- **read_documents.py**
  - generate the corpus database: `aviation_corpus.pkl`
- **aviation_chunk_saver.py**
  - generate the chunks to be processed into embeddings
- **extract_pkl_to_json.py**
  - save aviation_corpus.pkl into a json file aviation_corpus.json
- **check_pkl_content.py**
  - check the pkl content for documents - simple check
- **check_new_chunks.js**
  - check if is there new chunks to generate the embeddings or Not if Yes call `generate_embeddings.js`
- **check_embeddings.py**
  - check the new embeddings - cross check if have new embeddings store in AstrDB, if not close
- **check_astradb_content.js**
  - check AstraDB table to verify the embeddings and files stored
- **check_astradb_consistency**
  - cross check between AstraDB content and local embeddings files to verify the files consistency
- **visualizing-data.py**
  - generate local charts with data from the aviation_corpus.json and save aviation_corpus.json update with indicators.

### **2. Web Scraping Integration Plan**

#### **Objective**

To integrate web scraping into the AviationRAG pipeline without disrupting the current document processing workflow.
The goal is to extract aviation-related content from specific webpages, structure it into the same chunk format as DOCX documents,
and merge both sources before generating embeddings.

#### **Approach**

- **Maintain `read_documents.py` as-is**: It will continue handling DOCX files without modification.
  - This script generate the aviation_corpus.pkl:

- **Develop `scraping_documents.py` as a dedicated web scraping module**
  - This script shall be able to generate a file with the same content as 'read_documents.py' to append to 'aviation_corpus.pkl'
  - This script will scrape aviation-related webpages, extract relevant text, and process it into structured chunks.
  - The generated chunks will follow the same format as the DOCX-based chunks to ensure compatibility.
  - the file generate by this script shall be the following format.
  
```python
      {
       'filename': filename,
       'text': preprocessed_text,
       'tokens': lemmatized_tokens,
       'section_references': section_references,  # ✅ Ensure this is included
       'personal_names': personal_names,
       'entities': entities,
       'pos_tags': pos_tags,
       'metadata': metadata,
      'category': document_category
    }
```

### **3. Implementation Steps**

#### **Step 1: Define the web scrapping Format**

- Ensure web-scraped content follow this standardized structure: aviation_corpus.pkl
  
  ```python
  {
       'filename': filename,
       'text': preprocessed_text,
       'tokens': lemmatized_tokens,
       'section_references': section_references,  # ✅ Ensure this is included
       'personal_names': personal_names,
       'entities': entities,
       'pos_tags': pos_tags,
       'metadata': metadata,
      'category': document_category
    }
  ```

- Including `"source_type"` will help distinguish between different content origins for future analytics. ???

#### **Step 2: Develop `scrapping_documents.py`**

- Implement **BeautifulSoup** for static webpages or **Selenium** for dynamic content.
- Extract only **relevant sections** (ignoring ads, headers, footers, and navigation bars).
- Process the scraped content into coherent **chunks**, ensuring consistent length with DOCX-based ones.
- Save the output in `data/processed/chunked_web_documents/`.
- Generate the `aviation_corpus-scrapping.pkl` save in the `data/raw/aviation_corpus_scrapping.pkl`

#### **Step 3: Merge Both aviation_corpus.pkl Sources**

- Append new aviation_corpus_scrapping.pkl before running aviation_chunk_saver.py and embedding generation process.
- Implement duplicate handling to prevent redundant data from being stored.

#### **Step 4: Continue Normal Processing**

- Once chunks are created, embedding generation and storage in AstraDB will proceed as usual.
- FAISS-based search will now retrieve information from **both structured documents and real-time web data**.

### **4. Future Refinements & Considerations**

- **Fine-tune FAISS parameters** to optimize similarity search results.
- **Enhance `aviationai.py`** for better responses once FAISS testing is complete.
- **Expand Web Data Sources** by selecting high-quality aviation websites (e.g., regulatory bodies, aerospace industry publications, and aviation safety blogs).
- **Monitor data freshness**: Implement scheduled web scraping to update and refresh information over time.

### **5. Next Steps**

✅ **List key aviation-related websites** to scrape.
✅ **Define scraping rules** (targeting news, regulations, industry insights).
✅ **Select appropriate scraping methods** (BeautifulSoup for static content, Selenium for JavaScript-heavy pages).

Once these steps are defined, development of `scrape_and_chunk.py` can begin, making AviationRAG an even more powerful tool for aviation knowledge retrieval. 🚀✈️
