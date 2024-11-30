# Aviation RAG Projec

## Aviation RAG Purpose

### Aeronautical community - aeronautical industry is complex, since the one simple idea until design, manufacture, assembly, and fly there is a long journey, today with the industry demanding connection between different countries, resources and using the electronic connection to keep this system working accurately. The amount of data, information, and decision, that have to be taken since the initial project stages and final release are immense, and obviously the decisions has a direct impact on safety. Safety in aviation is a behavior, culture, safety must be part of aviation thinking, in this complex world of parts, subparts, systems, subsystems, connections, wires, requirements, and regulations, cultivating a strong safety culture is essential. In the midst of this myriad of data, processes, information, the decision has to be taken, anyhow. Recently with development of LLM - Large Language Models - the capability to address large amount the information has increasing significantly. LLM has been applied in different areas, such as healthcare, education and training, cybersecurity, improve decision-making, finance, military, human resource. LLM is not simply a tool for marketing and sell things. LLM can decipher our language on a huge scale, these models using training processes, absorb an incredible volume of texts to predict sentences. LLM with its versality of applications is changing the way, it can boost the human productivity in strategic thinking where the analysis from different areas collide such in aviation industry

## Aviation RAG Project Composition

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
    