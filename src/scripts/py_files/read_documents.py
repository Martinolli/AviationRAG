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

from config import (
    ABBREVIATIONS_CSV,
    DOCUMENTS_DIR,
    LOG_DIR,
    PKL_FILENAME,
    TEXT_EXPANDED_DIR,
    TEXT_OUTPUT_DIR,
)

# Load spaCy's English model
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2000000  # or any other suitable value

# Initialize spellchecker
spell = SpellChecker()

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", message="usetex mode requires TeX.")

# Global stopwords
STOP_WORDS = set(stopwords.words('english'))
import docx

log_file_path = LOG_DIR / "read_documents.log"

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging properly
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG to capture everything
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),  # Ensure file is written
        logging.StreamHandler()  # Print logs to console as well
    ]
)

logging.info("Logging initialized successfully.")

# Ensure directories exist
for directory in [TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR, PKL_FILENAME.parent]:
    os.makedirs(directory, exist_ok=True)  # No need to log this for every run

# Donwload NLTK data
def download_nltk_data():
    """Downloads required NLTK data only if not already installed."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_data()

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
        with open(ABBREVIATIONS_CSV, mode='r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            abbreviation_dict = {rows[0].strip(): rows[1].strip() for rows in reader if len(rows) >= 2}
    except FileNotFoundError:
        logging.error(f"Error: The file '{ABBREVIATIONS_CSV}' was not found.")
    except Exception as e:
        logging.exception("An error occurred while loading the abbreviation dictionary.")
    return abbreviation_dict

def split_connected_words_improved(text):
    words = re.findall(r'\w+|\W+', text)
    split_words = []
    
    for word in words:
        # ✅ Keep section references like "4.1.2", "§ 5.1" intact
        if re.match(r'^\d+(\.\d+)+$', word) or re.match(r'^§\s*\d+(\.\d+)*$', word):
            split_words.append(word)
            continue  

        # ✅ Keep "Level 1", "Part 91"
        if re.match(r'^[A-Za-z]+\s\d+$', word):  
            split_words.append(word)
            continue  

        # ✅ Keep aviation model numbers (e.g., "A320-214") intact
        if re.match(r'^[A-Za-z]+\d+[-]?\d*$', word):
            split_words.append(word)
            continue  

        # ✅ Process long alphanumeric words but keep numbers
        if len(word) > 15 and word.isalnum():
            split_parts = re.findall('[A-Z][a-z]*|[a-z]+|[0-9]+', word)
            split_words.extend(split_parts)
        else:
            split_words.append(word)

    # Apply wordninja ONLY to words, NOT numbers
    processed_text = ' '.join(split_words)
    processed_text = ' '.join(wordninja.split(processed_text))

    return processed_text
    
def filter_non_sense_strings(text):
    words = text.split()
    cleaned_words = []
    for word in words:
        # Keep numbers and section references like "5.1.2" or "Part 5"
        if re.match(r'^[a-zA-Z0-9.\-]+$', word) and len(set(word.lower())) > 2:
            cleaned_words.append(word)
    return ' '.join(cleaned_words)

def extract_section_references(text):
    """Extract structured numerical references like '4.1.2' and 'Level 1'."""
    import re

    # Updated regex patterns
    section_pattern = re.findall(r'\b\d+(\.\d+)+\b', text)  # Matches "4.1.2"
    level_pattern = re.findall(r'\b[Ll]evel\s\d+\b', text)  # Matches "Level 1"
    part_pattern = re.findall(r'\b[Pp]art\s\d+\b', text)  # Matches "Part 121"
    paragraph_pattern = re.findall(r'^§\s*\d+(\.\d+)*$', text)  # Matches "§ 5.1"

    references = section_pattern + level_pattern + part_pattern + paragraph_pattern

    # Debugging output
    if references:
        logging.info(f"Extracted Section References: {references}")
    else:
        logging.warning(f"No section references found in text.")

    return references if references else ["No References Found"]

def preprocess_text_with_sentences(text):
    doc = nlp(text)
    sentences = []
    
    for sent in doc.sents:
        cleaned_sentence = ' '.join(
            token.lemma_.lower() for token in sent
            if (token.is_alpha or token.like_num or re.match(r'^\d+(\.\d+)*$', token.text))  # Preserve numbers
            and token.text.lower() not in STOP_WORDS
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
    # References as entities
    section_references = extract_section_references(text)
    for ref in section_references:
        entities.append((ref, "SECTION_REF"))
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
    
    # Debug: Check if we have any texts
    logging.debug(f"Number of documents: {len(texts)}")
    if len(texts) == 0:
        logging.error("No texts found in documents")
        return

    # Debug: Check the content of the first few texts
    for i, text in enumerate(texts[:5]):
        logging.debug(f"Text {i} (first 100 chars): {text[:100]}")
        logging.debug(f"Text {i} length: {len(text)}")

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
    except ValueError as e:
        logging.error(f"ValueError in TfidfVectorizer: {e}")
        logging.error("Vocabulary: " + str(vectorizer.vocabulary_))
        return

def extract_metadata(file_path):
    """
    Extract metadata from PDF or DOCX files.
    
    Args:
        file_path (str): Path to the file.

    Returns:
        dict: Metadata extracted from the file.
    """
    metadata = {
        'title': '',
        'author(s)': '',
        'category': '',
        'tags': ''
    }
    
    # Handle PDF files
    if file_path.endswith('.pdf'):
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pdf_metadata = reader.metadata or {}
            metadata['title'] = pdf_metadata.get('/Title', '')
            metadata['author(s)'] = pdf_metadata.get('/Author', '')
    
    elif file_path.endswith('.docx'):
        try:
            doc = docx.Document(file_path)
            # Look at the first few paragraphs for metadata
            text = '\n'.join([p.text for p in doc.paragraphs[:5]])
            
            # Use regex to find metadata
            title_match = re.search(r'Title:\s*(.*)', text)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
            
            author_match = re.search(r'Author\(s\):\s*(.*)', text)
            if author_match:
                metadata['author(s)'] = author_match.group(1).strip()
            
            category_match = re.search(r'Category:\s*(.*)', text)
            if category_match:
                metadata['category'] = category_match.group(1).strip()
            
            tags_match = re.search(r'Tags:\s*(.*)', text)
            if tags_match:
                metadata['tags'] = tags_match.group(1).strip()
            
        except Exception as e:
            logging.info(f"Error reading DOCX metadata: {e}")
    
    return metadata

def classify_document(text):
    keywords = {
        'safety': [
            'safety', 'hazard', 'risk', 'incident', 'accident',
             'system', 'emergency', 'prevention',
            'safety management', 'safety culture', 'safety audit',
            'safety inspection', 'safety compliance', 'safety training'
        ],
        'maintenance': [
            'maintenance', 'repair', 'overhaul', 'inspection', 'servicing',
            'maintenance schedule', 'maintenance manual', 
            'maintenance log', 'maintenance record',
            'maintenance check', 'maintenance procedure',
            'maintenance crew', 'maintenance facility'
        ],
        'operations': [
            'flight', 'takeoff', 'landing', 'crew', 'pilot', 'aircraft', 'airplane',
            'operations manual', 'flight operations',
            'flight plan', 'flight schedule', 'flight crew', 'flight deck',
            'flight control', 'air traffic control', 'ground operations'
        ],
        'regulations': [
            'regulation', 'compliance', 'standard', 'rule', 'law', 'FAA',
            'ICAO', 'EASA', 'aviation regulation', 'aviation law',
            'aviation compliance', 'aviation standard', 'aviation rule',
            'regulatory requirement', 'regulatory compliance'
        ],
        'quality': [
            'quality', 'performance', 'service', 'customer', 'satisfaction',
            'design', 'quality assurance', 'quality control',
            'quality management', 'quality audit', 'quality inspection',
            'quality standard', 'quality improvement', 'quality system', 'quality policy'
        ],
        'accident_reports': [
            'accident', 'crash', 'collision', 'investigation', 'report',
            'NTSB', 'FAA', 'safety board', 'incident', 'fatality',
            'injury', 'damage', 'wreckage', 'black box', 'flight recorder',
            'accident analysis', 'accident summary', 'accident investigation',
            'accident report', 'accident findings', 'accident cause', 'accident prevention'
        ],
        'training_education': [
            'training', 'education', 'certification', 'course', 'syllabus',
            'training program', 'instructor', 'trainee', 'aviation school',
            'flight school', 'training manual'
        ],
        'weather_environment': [
            'weather', 'environment', 'climate', 'turbulence', 'storm',
            'wind', 'visibility', 'meteorology', 'weather report',
            'weather forecast', 'environmental impact'
        ],
        'technology_innovation': [
            'technology', 'innovation', 'avionics', 'automation', 'AI',
            'artificial intelligence', 'drone', 'UAV', 'unmanned aerial vehicle',
            'new technology', 'technological advancement'
        ],
        'security': [
            'security', 'threat', 'terrorism',
            'hijacking', 'security measures',
            'airport security', 'security protocol',
            'security breach', 'security incident', 'cybersecurity'
        ],
        'finance_economics': [
            'finance', 'economics', 'cost', 'budget',
            'funding', 'investment',
            'economic impact', 'financial report',
            'financial analysis', 'revenue', 'expense', 'profit', 'loss'
        ],
        'human_factors': [
            'human factors', 'ergonomics', 'fatigue',
            'stress', 'workload',
            'human performance', 'crew resource management',
            'CRM', 'human error', 'human-machine interaction'
        ],
        'emergency_response': [
            'emergency', 'response', 'rescue',
            'evacuation', 'emergency procedures',
            'emergency landing', 'emergency services',
            'first aid', 'crisis management', 'disaster response'
        ],
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

        logging.info(f"Processing {filename}...")

        expanded_text = expand_abbreviations_in_text(text, abbreviation_dict)

        raw_text = expanded_text

        # Debug BEFORE preprocessing
        logging.debug(f"Raw expanded text BEFORE processing: {expanded_text[:500]}")

        expanded_text = split_connected_words_improved(expanded_text)
        
        expanded_text = filter_non_sense_strings(expanded_text)

        # Debug AFTER preprocessing
        logging.debug(f"Expanded text AFTER preprocessing: {expanded_text[:500]}")

        section_references = extract_section_references(expanded_text)

        if not section_references:
            logging.warning(f"No section references found in {filename}")
        
        preprocessed_text = preprocess_text_with_sentences(expanded_text)

        personal_names = extract_personal_names(preprocessed_text)

        entities, pos_tags = extract_entities_and_pos_tags(preprocessed_text)

        tokens = [token.text.lower() for token in nlp(preprocessed_text) 
          if (token.is_alpha or token.like_num or re.match(r'^\d+(\.\d+)*$', token.text) or re.match(r'^[A-Za-z]+\s\d+$', token.text) or re.match(r'^§\s*\d+(\.\d+)*$', token.text))]

        tokens_without_stopwords = [token for token in tokens if token not in STOP_WORDS]

        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens_without_stopwords]

        if text_expanded_dir:
            # Remove .docx extension before adding .txt
            clean_filename = filename.replace('.docx', '')
            output_file_path = os.path.join(text_expanded_dir, f'{clean_filename}.txt')
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(raw_text)
            logging.info(f"Expanded text saved to: {output_file_path}")
            logging.info(f"Expanded text saved to: {output_file_path}")
               
        if text_output_dir:
            # Remove .docx extension before adding .txt
            clean_filename = filename.replace('.docx', '')
            output_file_path = os.path.join(text_output_dir, f'{clean_filename}.txt')
            logging.info(f"Processed text saved to: {output_file_path}")
            with open(output_file_path, 'w', encoding='utf-8') as out_file:
                out_file.write(preprocessed_text)
            logging.info(f"Text saved to: {output_file_path}")

        logging.info(f"Finished processing all documents in {directory_path}")
        metadata = extract_metadata(file_path)
        document_category = classify_document(preprocessed_text)

        # Remove 'category' field if it already exists
        if 'category' in metadata:
            del metadata['category']

        new_documents.append({
            'filename': filename,
            'text': preprocessed_text,
            'tokens': lemmatized_tokens,
            'section_references': section_references,  # ✅ Ensure this is included
            'personal_names': personal_names,
            'entities': entities,
            'pos_tags': pos_tags,
            'metadata': metadata,
            'category': document_category
        })

        # Debugging output
        logging.info(f"Stored Section References for {filename}: {section_references}")

    return existing_documents + new_documents

def main():
    try:
        documents = None
        if os.path.exists(PKL_FILENAME):
            with open(PKL_FILENAME, 'rb') as file:
                documents = pickle.load(file)
            
        if documents is None:
            logging.info("Reading documents from directory...")
            documents = read_documents_from_directory(DOCUMENTS_DIR, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR)
        else:
            logging.info("Appending new documents to the existing list...")
            documents = read_documents_from_directory(DOCUMENTS_DIR, TEXT_OUTPUT_DIR, TEXT_EXPANDED_DIR, documents)

        # Debug: Check documents before keyword extraction
            logging.debug(f"Number of documents before keyword extraction: {len(documents)}")
            for i, doc in enumerate(documents[:5]):
                logging.debug(f"Document {i} text length: {len(doc['text'])}")


        # Apply keyword extraction
        extract_keywords(documents)

        # Download NLTK data
        download_nltk_data()

        # Save the updated list
        with open(PKL_FILENAME, 'wb') as file:
            pickle.dump(documents, file)

        logging.info(f"Total documents: {len(documents)}")
    except Exception as e:
        logging.exception("An error occurred in main:")
        logging.info(f"An error occurred: {e}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logging.info("Starting document processing script")
    main()
    logging.info("Document processing script completed")

