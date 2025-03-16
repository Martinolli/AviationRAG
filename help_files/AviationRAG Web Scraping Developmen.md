# Aviation Scraping Plan

AviationRAG Web Scraping Development Plan

## **1. Current Status of the AviationRAG Project**

- **Pipeline Implementation**: `aviationrag_manager.py` is effectively managing the document processing pipeline.
- **Chat System Stability**: `aviationai.py` is functional, and while future adaptations may be needed, it is currently working well.
- **Search Optimization**: FAISS has been implemented for efficient and comprehensive similarity search, significantly improving retrieval speed and accuracy.
- **Testing Phase**: FAISS is undergoing tests to validate its performance with the stored embeddings.

### **2. Web Scraping Integration Plan**

#### **Objective**

To integrate web scraping into the AviationRAG pipeline without disrupting the current document processing workflow.
The goal is to extract aviation-related content from specific webpages, structure it into the same chunk format as DOCX documents,
and merge both sources before generating embeddings.

#### **Approach**

- **Maintain `read_documents.py` as-is**: It will continue handling DOCX files without modification.
- **Develop `scrape_and_chunk.py` as a dedicated web scraping module**:
  - This script will scrape aviation-related webpages, extract relevant text, and process it into structured chunks.
  - The generated chunks will follow the same format as the DOCX-based chunks to ensure compatibility.
- **Merge the Two Sources at the Chunking Stage**:
  - Both document-based and web-scraped chunks will be consolidated into the same pipeline before embedding generation.
  - This ensures a unified approach, allowing seamless integration without additional pipeline modifications.

### **3. Implementation Steps**

#### **Step 1: Define the Chunk Format**

- Ensure both Word and web-scraped content follow this standardized structure:
  
  ```json
  {
    "chunk_id": "unique_id",
    "filename": "source_name",
    "text": "chunked_text",
    "metadata": {"source_type": "web" or "document", "date": "...", "url": "..."}
  }
  ```

- Including `"source_type"` will help distinguish between different content origins for future analytics.

#### **Step 2: Develop `scrape_and_chunk.py`**

- Implement **BeautifulSoup** for static webpages or **Selenium** for dynamic content.
- Extract only **relevant sections** (ignoring ads, headers, footers, and navigation bars).
- Process the scraped content into coherent **chunks**, ensuring consistent length with DOCX-based ones.
- Save the output in `data/processed/chunked_web_documents/`.

#### **Step 3: Merge Both Chunk Sources**

- Modify the pipeline to read chunks from both DOCX and web-scraped sources.
- Append new chunks before running them through the embedding generation process.
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
